"""v0.95.5 entry; preserve the frozen question, delivery and transport workflow."""
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import ask as frozen
from story_agent import default_v955 as default
from story_agent.wire_default_v955 import AuditedDeepSeek


def execute(*args,**kwargs):
    with patch.object(frozen,'default',default),patch.object(frozen,'AuditedDeepSeek',AuditedDeepSeek):
        return frozen.execute(*args,**kwargs)


def main(argv=None):
    with patch.object(frozen,'default',default),patch.object(frozen,'AuditedDeepSeek',AuditedDeepSeek):
        return frozen.main(argv)


if __name__=='__main__':sys.exit(main())
