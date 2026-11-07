import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import json
import re
from pathlib import Path
from llamafactory.chat import ChatModel

root = Path("/root/autodl-tmp/rank1-finetune")

print("正在加载基础模型和微调权重……", flush=True)
model = ChatModel({
    "model_name_or_path": "/root/autodl-tmp/models/Qwen2.5-1.5B",
    "adapter_name_or_path": str(root / "outputs/qwen_1_5b_rank1_lora"),
    "finetuning_type": "lora",
    "template": "empty",
    "add_special_tokens": "<think>,</think>",
    "resize_vocab": True,
    "trust_remote_code": False,
    "infer_dtype": "bfloat16",
    "flash_attn": "sdpa",
})

rows = json.loads((root / "data/val_500.json").read_text())
results = []

def extract_label(text):
    match = re.search(r"</think>\s*(true|false)\b", text)
    return match.group(1) if match else None

for i, row in enumerate(rows[:3], start=1):
    prompt = "\n".join(
        part for part in [row["instruction"], row["input"]] if part
    )

    print(f"\n{'=' * 50}\n样本 {i}", flush=True)
    print(row["input"], flush=True)
    print("\n正在生成回答……", flush=True)

    response = model.chat(
        [{"role": "user", "content": prompt}],
        do_sample=False,
        max_new_tokens=768,
        skip_special_tokens=False,
    )[0]

    text = response.response_text
    expected = extract_label(row["output"])
    predicted = extract_label(text)

    print("\n模型实际输出：\n" + text, flush=True)
    print(f"\n参考标签：{expected}", flush=True)
    print(f"提取的预测标签：{predicted}", flush=True)
    print(f"结束原因：{response.finish_reason}", flush=True)

    results.append({
        "sample_index": i - 1,
        "input": row["input"],
        "expected": expected,
        "predicted": predicted,
        "generated_text": text,
        "finish_reason": response.finish_reason,
    })

path = root / "logs/generation_examples.json"
path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
print(f"\n测试输出已保存：{path}", flush=True)
