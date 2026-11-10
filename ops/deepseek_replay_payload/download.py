import concurrent.futures
import hashlib
import json
import sys
import time
from pathlib import Path
from huggingface_hub import snapshot_download
from evidence import save

root = Path(__file__).resolve().parent
which = sys.argv[1]
if which == 'model':
    snapshot_download('deepseek-ai/DeepSeek-V4.1-Flash-1017', revision='1bae0e1088993b9b82de08f59de5be5a34e42170', local_dir='/workspace/model', max_workers=16)
    manifest = json.loads((root / 'model-files.json').read_text())
    files = [x for x in manifest if x['type'] == 'file' and x.get('lfs')]
    started = time.monotonic()
    def check(item):
        p = Path('/workspace/model') / item['path']
        assert p.stat().st_size == item['size'], str(p)
        digest = hashlib.sha256()
        with p.open('rb') as stream:
            while block := stream.read(16 * 1024 * 1024):
                digest.update(block)
        assert digest.hexdigest() == item['lfs']['oid'], 'Shard hash mismatch: ' + str(p)
        return {'path': item['path'], 'sha256': digest.hexdigest(), 'size': item['size']}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        checked = list(pool.map(check, files))
    assert checked and sum(x['size'] for x in checked) > 400_000_000_000
    save(root / 'evidence/model-integrity.json', {'status': 'all_lfs_files_verified', 'revision': '1bae0e1088993b9b82de08f59de5be5a34e42170', 'hash_seconds': time.monotonic() - started, 'files': checked})
else:
    snapshot_download('inferencekey/deepinfer-sm100-cu132', revision='6c4ae0ee7c49502a09241faf4abbc515ce4fe17a', local_dir='/workspace/prebuilt', max_workers=8)
    manifest = json.loads((root / 'prebuilt-manifest.json').read_text())
    for item in manifest:
        if item['type'] == 'file' and item.get('lfs'):
            p = Path('/workspace/prebuilt') / item['path']
            assert hashlib.sha256(p.read_bytes()).hexdigest() == item['lfs']['oid'], str(p)
    save(root / 'evidence/prebuilt-integrity.json', {'status': 'all_lfs_files_verified', 'revision': '6c4ae0ee7c49502a09241faf4abbc515ce4fe17a'})
