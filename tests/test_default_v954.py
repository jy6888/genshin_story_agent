"""Mutable summary lock recovery; no API calls."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts import ask_v954


class SummaryLockTest(unittest.TestCase):
    def setUp(self):
        self.tmp_root=Path(__file__).resolve().parents[1]/'.runtime/tmp-v954'
        self.tmp_root.mkdir(parents=True,exist_ok=True)

    def test_transient_permission_error_retries_without_changing_payload(self):
        with tempfile.TemporaryDirectory(dir=self.tmp_root) as tmp:
            p=Path(tmp)/'cost.json';p.write_text('{"old":true}',encoding='utf-8')
            original=ask_v954.os.replace;attempts=[]
            def replace(a,b):
                attempts.append((a,b))
                if len(attempts)==1:raise PermissionError(5,'Access is denied')
                return original(a,b)
            with patch.object(ask_v954.os,'replace',side_effect=replace),patch.object(ask_v954.time,'sleep') as sleeper:
                ask_v954.replace_json(p,{'中文':'完整摘要'})
            self.assertEqual(json.loads(p.read_text(encoding='utf-8')),{'中文':'完整摘要'})
            self.assertEqual(sleeper.call_args_list[0].args,(.02,))
            self.assertEqual(len(attempts),2);self.assertEqual(list(Path(tmp).glob('*.tmp')),[])

    def test_persistent_permission_stops_and_preserves_previous_summary(self):
        with tempfile.TemporaryDirectory(dir=self.tmp_root) as tmp:
            p=Path(tmp)/'cost.json';p.write_text('{"old":true}',encoding='utf-8')
            with patch.object(ask_v954.os,'replace',side_effect=PermissionError(5,'Access is denied')) as replace,patch.object(ask_v954.time,'sleep'):
                with self.assertRaises(PermissionError):ask_v954.replace_json(p,{'new':True})
            self.assertEqual(replace.call_count,8)
            self.assertEqual(json.loads(p.read_text(encoding='utf-8')),{'old':True})
            self.assertEqual(list(Path(tmp).glob('*.tmp')),[])
