import hashlib
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from . import audit_model as a


class ModelIntegrityTests(unittest.TestCase):
    def test_missing_extra_and_changed_shards_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'first.gguf';path.write_bytes(b'weights')
            manifest={'kind':'gguf','files':[{'name':'first.gguf','bytes':7,'sha256':hashlib.sha256(b'weights').hexdigest()}]}
            with self.assertRaises(ValueError):a.verify_files(manifest,root)
            with patch.dict(a.c.CANDIDATE['weight_inventory_sha256'],{'gguf':a.c.digest(manifest)}):
                result=a.verify_files(manifest,root)
                self.assertFalse(result['conversion_fidelity_audited']);self.assertFalse(result['publication_allowed'])
                path.write_bytes(b'changed')
                with self.assertRaises(ValueError):a.verify_files(manifest,root)
                path.unlink()
                with self.assertRaises(ValueError):a.verify_files(manifest,root)
                path.write_bytes(b'weights');(root/'extra.gguf').write_bytes(b'extra')
                with self.assertRaises(ValueError):a.verify_files(manifest,root)


if __name__=='__main__':unittest.main()
