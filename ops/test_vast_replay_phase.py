import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('phase',Path(__file__).with_name('vast_replay_phase.py'))
phase=importlib.util.module_from_spec(spec);spec.loader.exec_module(phase)


class OwnershipTests(unittest.TestCase):
    def test_inventory_only_selects_exact_owned_label(self):
        with patch.object(phase,'api',return_value={'instances':[{'id':1,'label':'owned'},{'id':2,'label':'other'}]}):
            self.assertEqual(phase.owned({'label':'owned'}),[{'id':1,'label':'owned'}])

    def test_bad_inventory_does_not_mean_gone(self):
        with patch.object(phase,'api',return_value={}):
            with self.assertRaises(ValueError):phase.owned({'label':'owned'})

    def test_no_launch_after_attempt_or_above_ceiling(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'phase.json').write_text(json.dumps({'deadline':time.time()+3600,'hours':1,
                'quote':{'dph_total':10},'network_reserve_usd':1,'spend_ceiling_usd':5}))
            with patch.object(phase,'api') as api:
                with self.assertRaises(ValueError):phase.launch(root)
                api.assert_not_called()
                (root/'launch-intent.json').write_text('{}')
                with self.assertRaises(ValueError):phase.launch(root)
                api.assert_not_called()


if __name__=='__main__':unittest.main()
