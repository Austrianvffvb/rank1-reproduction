"""Download the original released JSON; verify against the experiment manifest."""
import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--filename', help='Exact JSON path from the official dataset repository')
parser.add_argument('--revision', default='main', help='Use a pinned dataset commit when available')
args = parser.parse_args()
repo = 'jhu-clsp/rank1-training-data'
expected = json.loads((root/'data/data_manifest.json').read_text())['source_sha256']
api = HfApi()
revision = api.dataset_info(repo, revision=args.revision).sha
files = api.list_repo_files(repo, repo_type='dataset', revision=revision)
candidates = [args.filename] if args.filename else [p for p in files if p.endswith('.json')]
if len(candidates) != 1:
    raise RuntimeError(f'Cannot uniquely identify original file. Inspect dataset repository. Candidates: {candidates}')
path = Path(hf_hub_download(repo, candidates[0], repo_type='dataset', revision=revision))
digest = hashlib.sha256(path.read_bytes()).hexdigest()
if digest != expected:
    raise RuntimeError(f'Dataset changed: expected {expected}, got {digest}. Do not overwrite experiment provenance.')
target = root/'data/rank1_official_train.json'
target.write_bytes(path.read_bytes())
(root/'data/download_manifest.json').write_text(json.dumps({'repo_id':repo, 'revision':revision, 'file':candidates[0], 'sha256':digest}, indent=2))
print(target)
