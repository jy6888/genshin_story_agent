"""Portable configuration and no-network missing-data regressions."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from story_agent import strong_agent_sa2 as config
from story_agent.budget import BudgetStopped
from story_agent.public_data import require_local_data
from scripts import ask


class PublicReleaseTest(unittest.TestCase):
    def test_credentials_precedence_and_repository_dotenv(self):
        with tempfile.TemporaryDirectory(dir=config.ROOT/'.runtime') as temp:
            root = Path(temp)
            (root/'.env').write_text('OPENAI_BASE_URL=https://api.deepseek.com\nOPENAI_API_KEY=offline-file-placeholder\n', encoding='utf-8')
            with patch.object(config, 'ROOT', root), patch.dict(os.environ, {}, clear=True):
                self.assertEqual(config.credentials(), ('https://api.deepseek.com', 'offline-file-placeholder'))
                with patch.dict(os.environ, {'OPENAI_API_KEY': 'offline-env-placeholder'}):
                    self.assertEqual(config.credentials()[1], 'offline-env-placeholder')
                with patch.dict(os.environ, {'OPENAI_API_KEY': ''}):
                    with self.assertRaises(BudgetStopped): config.credentials()

    def test_endpoint_restrictions_and_missing_key(self):
        endpoints = ['http://api.deepseek.com', 'https://example.invalid',
                     'https://api.deepseek.com.example.invalid', 'https://user:password@api.deepseek.com',
                     'https://api.deepseek.com?key=value', 'https://api.deepseek.com#fragment']
        with patch.dict(os.environ, {}, clear=True), patch('dotenv.dotenv_values', return_value={}):
            with self.assertRaises(BudgetStopped): config.credentials()
            for endpoint in endpoints:
                with self.subTest(endpoint=endpoint), patch.dict(os.environ,
                        {'OPENAI_API_KEY': 'offline-env-placeholder', 'OPENAI_BASE_URL': endpoint}):
                    with self.assertRaises(BudgetStopped) as caught: config.credentials()
                    self.assertNotIn('offline-env-placeholder', str(caught.exception))

    def test_missing_assets_stop_before_credentials_model_or_output(self):
        with tempfile.TemporaryDirectory(dir=config.ROOT/'.runtime') as temp:
            root = Path(temp)
            with self.assertRaisesRegex(FileNotFoundError, 'Local research assets'):
                require_local_data(root)
            with patch.object(ask, 'ROOT', root), patch.object(ask, 'credentials') as credentials, \
                    patch.object(ask, 'live_model') as model:
                with self.assertRaisesRegex(FileNotFoundError, 'No model request was sent'):
                    ask.execute('synthetic question', root/'answer')
                credentials.assert_not_called(); model.assert_not_called()
                self.assertFalse((root/'answer').exists())


if __name__ == '__main__': unittest.main()
