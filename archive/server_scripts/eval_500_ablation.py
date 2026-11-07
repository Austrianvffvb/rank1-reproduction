import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import json
import random
import re
import time
from pathlib import Path

import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from llamafactory.chat import ChatModel

root = Path("/root/autodl-tmp/rank1-finetune")
mode = sys.argv[1]
assert mode in ["base", "finetuned", "label_only"]

# 两次运行用同样的随机种子，选出同样的 50 条样本。
rows = json.loads((root / "data/val_500.json").read_text())
indices = list(range(len(rows)))
output = root / f"logs/eval500_{mode}.jsonl"
if output.exists():
    raise RuntimeError(f"结果文件已存在，避免覆盖：{output}")

args = {
    "model_name_or_path": "/root/autodl-tmp/models/Qwen2.5-1.5B",
    "finetuning_type": "lora",
    "template": "empty",
    "trust_remote_code": False,
    "infer_dtype": "bfloat16",
    "flash_attn": "sdpa",
}
if mode in ["finetuned", "label_only"]:
    args.update({
        "adapter_name_or_path": str(root / ("outputs/qwen_1_5b_label_only" if mode == "label_only" else "outputs/qwen_1_5b_rank1_lora")),
        "add_special_tokens": "<think>,</think>",
        "resize_vocab": True,
    })

print(f"正在加载：{mode}", flush=True)
chat = ChatModel(args)
engine = chat.engine
tokenizer = engine.tokenizer

def label(text):
    # 接受推理后的标签，也接受基础模型直接回答 true/false。
    match = re.search(r"</think>\s*(true|false)\b", text, re.I)
    if match:
        return match.group(1).lower()
    match = re.fullmatch(r"\s*(true|false)\s*", text, re.I)
    return match.group(1).lower() if match else None

class StopAfterLabel(StoppingCriteria):
    def __init__(self, prompt_length):
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores, **kwargs):
        text = tokenizer.decode(
            input_ids[0, self.prompt_length:],
            skip_special_tokens=False,
        )
        return label(text) is not None

results = []
with output.open("x", encoding="utf-8") as file:
    for position, index in enumerate(indices, 1):
        row = rows[index]
        expected = label(row["output"])
        assert expected is not None, f"参考标签无法解析：{index}"

        prompt = "\n".join(
            x for x in [row["instruction"], row["input"]] if x
        )
        prompt_ids, _ = engine.template.encode_oneturn(
            tokenizer,
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": ""},
            ],
        )
        inputs = torch.tensor([prompt_ids], device=engine.model.device)

        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            generated = engine.model.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                do_sample=False,
                max_new_tokens=768,
                eos_token_id=engine.template.get_stop_token_ids(tokenizer),
                pad_token_id=tokenizer.pad_token_id,
                stopping_criteria=StoppingCriteriaList([
                    StopAfterLabel(len(prompt_ids))
                ]),
            )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start

        ids = generated[0, len(prompt_ids):]
        text = tokenizer.decode(ids, skip_special_tokens=False)
        # 解析时去掉末尾 EOS/PAD，保留 think 标签。
        parsed_text = text
        for token in [tokenizer.eos_token, tokenizer.pad_token]:
            if token:
                parsed_text = parsed_text.replace(token, "")
        predicted = label(parsed_text)

        result = {
            "validation_index": index,
            "expected": expected,
            "predicted": predicted,
            "correct": predicted == expected,
            "generated_tokens": len(ids),
            "seconds": seconds,
            "finish_reason": (
                "label_complete" if predicted is not None
                else "length" if len(ids) >= 768 else "eos"
            ),
            "generated_text": text,
        }
        results.append(result)
        file.write(json.dumps(result, ensure_ascii=False) + "\n")
        file.flush()

        print(
            f"[{mode}] {position}/{len(indices)} "
            f"参考={expected} 预测={predicted} "
            f"正确={result['correct']} 耗时={seconds:.1f}s",
            flush=True,
        )

summary = {
    "model": mode,
    "samples": len(results),
    "correct": sum(r["correct"] for r in results),
    "accuracy": sum(r["correct"] for r in results) / len(results),
    "valid_label_rate": sum(r["predicted"] is not None for r in results) / len(results),
    "mean_seconds": sum(r["seconds"] for r in results) / len(results),
    "reference_true": sum(r["expected"] == "true" for r in results),
    "reference_false": sum(r["expected"] == "false" for r in results),
}
(root / f"logs/eval500_{mode}_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2)
)
print("\n汇总：", json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
