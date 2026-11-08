"""v0.95.3 adapter for the unchanged scripts.ask entry and output contract."""
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import ask as frozen
from story_agent import default_v953 as default
from story_agent.wire_default_v953 import AuditedDeepSeek


def execute(*args, **kwargs):
    with patch.object(frozen, 'default', default), patch.object(frozen, 'AuditedDeepSeek', AuditedDeepSeek):
        return frozen.execute(*args, **kwargs)


def main(argv=None):
    with patch.object(frozen, 'default', default), patch.object(frozen, 'AuditedDeepSeek', AuditedDeepSeek):
        return frozen.main(argv)


if __name__ == '__main__':
    sys.exit(main())
