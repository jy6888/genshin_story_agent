"""Default-v1.3: only the private conflict return boundary changes."""
from unittest.mock import patch
from story_agent import default_v953 as previous
from story_agent.conflict_check_default_v955 import ConflictCheck
from scripts.ask_v954 import replace_json,WAITS

VERSION='v0.95.5'
CONFIGURATION='default-v1.3'
ROOT,GRAPH,MODEL,SYSTEM_PROMPT=previous.ROOT,previous.GRAPH,previous.MODEL,previous.SYSTEM_PROMPT
AgentTools=previous.AgentTools
digest,same=previous.digest,previous.same


def configuration():
    return dict(previous.configuration(),configuration=CONFIGURATION,code_version=VERSION,
        conflict_source_same_file_multiple_locations=True,conflict_format_feedback_limit=1,
        conflict_privacy_rejections_feedback=False,mutable_json_permission_retry_seconds=list(WAITS),
        wrapper_hashes={'scripts/ask_v955.py':digest((ROOT/'scripts/ask_v955.py').read_text(encoding='utf-8')),
                       'scripts/ask_v954.py':digest((ROOT/'scripts/ask_v954.py').read_text(encoding='utf-8'))})


def build(*args,**kwargs):
    with patch.object(previous.frozen,'ConflictCheck',ConflictCheck):
        return previous.build(*args,**kwargs)


def finish(*args,**kwargs):
    with patch.object(previous,'CONFIGURATION',CONFIGURATION),patch.object(previous,'VERSION',VERSION):
        return previous.finish(*args,**kwargs)
