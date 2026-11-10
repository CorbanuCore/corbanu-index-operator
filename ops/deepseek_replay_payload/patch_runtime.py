import ast
import json
import sys
from pathlib import Path
from evidence import save, sha

root = Path(__file__).resolve().parent
source = Path(sys.argv[1]) / 'deepinfer/model_executor/models/v4_modules/engram_metadata.py'
text = source.read_text()
tree = ast.parse(text)
node = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'CompressedVocabulary')
expected_tokenizer = sha((root / 'tokenizer.json').read_bytes())
expected_map = sha((root / 'engram-map.json').read_bytes())
replacement = f'''class CompressedVocabulary:
    """Frozen, full-vocabulary parity-checked compression map; no runtime regex."""
    def __init__(self, tokenizer_path: str) -> None:
        import hashlib
        import json
        from pathlib import Path
        assert hashlib.sha256(Path(tokenizer_path).read_bytes()).hexdigest() == {expected_tokenizer!r}
        data = Path({str(root / 'engram-map.json')!r}).read_bytes()
        assert hashlib.sha256(data).hexdigest() == {expected_map!r}
        self.old2new = json.loads(data)
        self.lookup_table = np.asarray(self.old2new, dtype=np.int32)
        self.num_tokens = max(self.old2new) + 1

    def __len__(self) -> int:
        return self.num_tokens
'''
lines = text.splitlines(keepends=True)
patched = ''.join(lines[:node.lineno - 1]) + replacement + ''.join(lines[node.end_lineno:])
patched = patched.replace('from tokenizers import Regex, Tokenizer, normalizers\n', '')
assert 'Regex' not in patched
ast.parse(patched)
source.write_text(patched)
save(root / 'evidence/runtime-patch.json', {'file': str(source), 'before_sha256': sha(text.encode()), 'after_sha256': sha(patched.encode()), 'tokenizer_sha256': expected_tokenizer, 'static_map_sha256': expected_map})
