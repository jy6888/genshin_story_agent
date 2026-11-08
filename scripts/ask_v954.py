"""v0.95.4: tolerate brief Windows reader locks on mutable summaries only."""
import json
import os
from pathlib import Path
import sys
import time
import uuid
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import ask_v953
from story_agent import default_v953,wire_default_v953,wire_default_v1

VERSION='v0.95.4'
WAITS=(.02,.05,.1,.2,.4,.8,1.)
original_configuration=default_v953.configuration


def replace_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    try:
        for attempt in range(len(WAITS)+1):
            try:
                os.replace(temporary,path)
                return
            except PermissionError:
                if attempt==len(WAITS):raise
                time.sleep(WAITS[attempt])
    finally:
        if temporary.exists():temporary.unlink()


def configuration():
    return dict(original_configuration(),code_version=VERSION,
        mutable_json_permission_retry_seconds=list(WAITS),
        wrapper_hashes={'scripts/ask_v953.py':default_v953.digest((ROOT/'scripts/ask_v953.py').read_text(encoding='utf-8')),
                       'scripts/ask_v954.py':default_v953.digest(Path(__file__).read_text(encoding='utf-8'))})


def execute(*args,**kwargs):
    with patch.object(default_v953,'configuration',configuration),patch.object(default_v953,'VERSION',VERSION), \
         patch.object(wire_default_v953,'VERSION',VERSION), \
         patch.object(default_v953,'replace_json',replace_json), \
         patch.object(wire_default_v953,'replace_json',replace_json), \
         patch.object(wire_default_v1,'replace_json',replace_json):
        return ask_v953.execute(*args,**kwargs)


def main():
    with patch.object(default_v953,'configuration',configuration),patch.object(default_v953,'VERSION',VERSION), \
         patch.object(wire_default_v953,'VERSION',VERSION), \
         patch.object(default_v953,'replace_json',replace_json), \
         patch.object(wire_default_v953,'replace_json',replace_json), \
         patch.object(wire_default_v1,'replace_json',replace_json):
        return ask_v953.main()


if __name__=='__main__':sys.exit(main())
