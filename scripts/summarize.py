"""Audit saved predictions and regenerate summaries without GPU dependencies."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]

def query_key(text):
    match = re.match(r"^\s*Query:\s*(.*?)\r?\nPassage:", text, re.S)
    if not match:
        raise ValueError("Cannot extract query")
    return " ".join(match.group(1).split()).casefold()

def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n)) / denominator
    radius = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    return [center-radius, center+radius]

def quantile(values, p):
    values = sorted(values)
    pos = (len(values)-1)*p
    lower = math.floor(pos)
    upper = math.ceil(pos)
    return values[lower] + (values[upper]-values[lower])*(pos-lower)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=ROOT/"results/predictions")
    parser.add_argument("--validation", type=Path, default=ROOT/"data/val_500.json")
    parser.add_argument("--output", type=Path, default=ROOT/"results/verified_summary.json")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    all_rows = {}
    report = {"provenance": "Recomputed from archived per-pair predictions; not new model runs", "models": {}}
    for mode in ["base", "finetuned", "label_only"]:
        path = args.predictions/f"eval500_{mode}.jsonl"
        rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        indexed = {r["validation_index"]: r for r in rows}
        assert len(rows) == len(indexed) == 500
        assert set(indexed) == set(range(500))
        assert all(r["expected"] in ["true", "false"] for r in rows)
        assert all(r["predicted"] in ["true", "false", None] for r in rows)
        assert all(r["correct"] == (r["predicted"] == r["expected"]) for r in rows)
        correct = sum(r["correct"] for r in rows)
        tp = sum(r["expected"] == r["predicted"] == "true" for r in rows)
        fp = sum(r["expected"] == "false" and r["predicted"] == "true" for r in rows)
        positive = sum(r["expected"] == "true" for r in rows)
        stored = json.loads((args.predictions/f"eval500_{mode}_summary.json").read_text())
        assert stored["correct"] == correct and stored["samples"] == len(rows)
        assert math.isclose(stored["accuracy"], correct/len(rows))
        report["models"][mode] = {
            "samples": len(rows), "correct": correct, "accuracy": correct/len(rows),
            "valid_label_rate": sum(r["predicted"] is not None for r in rows)/len(rows),
            "precision_true": tp/(tp+fp), "recall_true": tp/positive,
            "mean_generated_tokens": statistics.mean(r["generated_tokens"] for r in rows),
            "historical_mean_seconds_not_cross_hardware_comparable": statistics.mean(r["seconds"] for r in rows),
            "pair_wilson_95_descriptive_only": wilson(correct, len(rows)),
            "prediction_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        all_rows[mode] = indexed
    assert all(all_rows[m][i]["expected"] == all_rows["base"][i]["expected"] for m in all_rows for i in range(500))
    a, b = all_rows["finetuned"], all_rows["label_only"]
    counts = Counter((a[i]["correct"], b[i]["correct"]) for i in range(500))
    report["paired_counts"] = {
        "both_correct": counts[True, True], "reasoning_only_correct": counts[True, False],
        "label_only_correct": counts[False, True], "both_wrong": counts[False, False],
    }
    report["interval_note"] = "Wilson intervals treat pairs as independent and are descriptive only. Query bootstrap accounts for shared queries, but not training-seed variation or validation-set adaptation."
    if args.validation.exists():
        data = json.loads(args.validation.read_text())
        assert len(data) == 500
        groups = defaultdict(list)
        for i, row in enumerate(data):
            expected = re.search(r"</think>\s*(true|false)\s*$", row["output"], re.I).group(1).lower()
            assert expected == a[i]["expected"]
            groups[query_key(row["input"])].append(i)
        report["unique_queries"] = len(groups)
        rng = random.Random(12345)
        clusters = list(groups.values())
        estimates = {m: [] for m in all_rows}
        differences = []
        for _ in range(args.bootstrap):
            ids = [i for group in rng.choices(clusters, k=len(clusters)) for i in group]
            acc = {m: sum(rows[i]["correct"] for i in ids)/len(ids) for m, rows in all_rows.items()}
            for m in estimates: estimates[m].append(acc[m])
            differences.append(acc["label_only"]-acc["finetuned"])
        for mode in estimates:
            report["models"][mode]["query_bootstrap_accuracy_95"] = [quantile(estimates[mode], p) for p in [.025, .975]]
        report["query_bootstrap_label_minus_reasoning_95"] = [quantile(differences, p) for p in [.025, .975]]
        report["bootstrap"] = {"replicates": args.bootstrap, "seed": 12345, "unit": "normalized query", "estimator": "pair-weighted accuracy"}
    else:
        report["query_bootstrap_status"] = "not computed: regenerate val_500.json first"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
