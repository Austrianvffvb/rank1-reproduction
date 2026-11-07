"""Derive label-only outputs without changing inputs or sample order."""
import json
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
for split in ['train_5000', 'val_500']:
    rows = json.loads((root/'data'/f'{split}.json').read_text())
    for row in rows:
        match = re.search(r'</think>\s*(true|false)\s*$', row['output'], re.I)
        if match is None:
            raise ValueError(f'Malformed reference output in {split}')
        row['output'] = match.group(1).lower()
    path = root/'data'/f'{split}_label_only.json'
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print(path, len(rows))
