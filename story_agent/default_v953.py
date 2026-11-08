"""Default-v1.2 composition, with the frozen v0.95.2 entry implementation."""
from unittest.mock import patch
from story_agent import default_v1 as frozen
from story_agent import report_default_v953 as report

VERSION = 'v0.95.3'
CONFIGURATION = 'default-v1.2'
ROOT, GRAPH, MODEL, SYSTEM_PROMPT = frozen.ROOT, frozen.GRAPH, frozen.MODEL, frozen.SYSTEM_PROMPT
AgentTools = frozen.AgentTools
digest, same, replace_json = frozen.digest, frozen.same, frozen.replace_json


def configuration():
    return dict(frozen.configuration(), code_version=VERSION, configuration=CONFIGURATION,
        complete_empty_reply_resends=1, complete_empty_reply_wait_seconds=0,
        wrapper_hashes={'scripts/ask_v953.py': digest((ROOT/'scripts/ask_v953.py').read_text(encoding='utf-8'))})


def build(model, directory, wiki, books=None):
    books = books if books is not None else AgentTools(ROOT)
    agent, session, checker, _ = frozen.build(model, directory, wiki, books)
    sources = report.Sources(books)
    checker.sources = sources
    return agent, session, checker, sources


def finish(*args, **kwargs):
    # Runtime injection reuses the exact frozen ordering, without editing its file.
    with patch.object(frozen, 'report', report), patch.object(frozen, 'VERSION', VERSION), \
            patch.object(frozen, 'CONFIGURATION', CONFIGURATION):
        return frozen.finish(*args, **kwargs)
