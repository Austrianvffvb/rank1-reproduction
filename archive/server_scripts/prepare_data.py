import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

SEED = 12345
TRAIN_SIZE = 5000
VAL_SIZE = 500

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/rank1_official_train.json"

def query_key(text):
    match = re.match(
        r"^\s*Query:\s*(.*?)\r?\nPassage:",
        text,
        flags=re.DOTALL,
    )
    if not match:
        return None
    # 忽略大小写和空白差异，减少同一查询跨集合的风险。
    key = " ".join(match.group(1).split()).casefold()
    return key or None

def get_label(output):
    match = re.search(
        r"</think>\s*(true|false)\s*$",
        output,
        flags=re.IGNORECASE,
    )
    return match.group(1).lower() if match else None

with SOURCE.open("r", encoding="utf-8") as f:
    raw = json.load(f)

valid = []
seen = set()
excluded = Counter()

for row in raw:
    fields = ("instruction", "input", "output")
    if not isinstance(row, dict) or not all(
        isinstance(row.get(k), str) for k in fields
    ):
        excluded["字段异常"] += 1
        continue

    key = query_key(row["input"])
    if key is None:
        excluded["无法提取查询"] += 1
        continue

    output = row["output"].strip()
    if not output.startswith("<think>") or get_label(output) is None:
        excluded["推理或标签格式异常"] += 1
        continue

    identity = (row["instruction"], row["input"], row["output"])
    if identity in seen:
        excluded["完全重复样本"] += 1
        continue
    seen.add(identity)

    # 原样保留三个训练字段，不改写教师的推理文本。
    valid.append((key, {k: row[k] for k in fields}))

# 先随机划分查询，再在各自的样本池中抽样。
keys = sorted({key for key, _ in valid})
rng = random.Random(SEED)
rng.shuffle(keys)

split_at = max(1, int(len(keys) * 0.1))
val_queries = set(keys[:split_at])

train_pool = [row for key, row in valid if key not in val_queries]
val_pool = [row for key, row in valid if key in val_queries]

if len(train_pool) < TRAIN_SIZE or len(val_pool) < VAL_SIZE:
    raise RuntimeError(
        f"可用样本不足：train={len(train_pool)}, val={len(val_pool)}"
    )

train = rng.sample(train_pool, TRAIN_SIZE)
val = rng.sample(val_pool, VAL_SIZE)

train_keys = {query_key(row["input"]) for row in train}
val_keys = {query_key(row["input"]) for row in val}
overlap = train_keys & val_keys
assert not overlap, "训练集与验证集存在查询重叠"

def stats(rows):
    return {
        "samples": len(rows),
        "unique_queries": len({query_key(r["input"]) for r in rows}),
        "labels": dict(Counter(get_label(r["output"]) for r in rows)),
    }

def save_json(name, content):
    path = ROOT / "data" / name
    with path.open("w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=2)
    return path

source_hash = hashlib.sha256()
with SOURCE.open("rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        source_hash.update(chunk)

manifest = {
    "seed": SEED,
    "source": SOURCE.name,
    "source_sha256": source_hash.hexdigest(),
    "source_rows": len(raw),
    "valid_rows": len(valid),
    "excluded": dict(excluded),
    "split_method": "query-disjoint; 10% of unique queries for validation pool",
    "train": stats(train),
    "validation": stats(val),
    "query_overlap": len(overlap),
}

save_json("train_5000.json", train)
save_json("val_500.json", val)
save_json("data_manifest.json", manifest)

print(json.dumps(manifest, ensure_ascii=False, indent=2))
print("\n已生成：")
print(ROOT / "data/train_5000.json")
print(ROOT / "data/val_500.json")
print(ROOT / "data/data_manifest.json")
