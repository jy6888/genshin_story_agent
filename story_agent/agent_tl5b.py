"""TL5b: one exact main-prompt insertion; all child surfaces unchanged."""
from deepagents import create_deep_agent
from deepagents.profiles import GeneralPurposeSubagentProfile, HarnessProfile, register_harness_profile
from story_agent import agent_tl5 as frozen
from story_agent.name_check_tl5b import RULE
from story_agent.report_tl5 import APPENDIX

VERSION = 'v0.94.0'
MODEL = frozen.MODEL
SYSTEM_PROMPT = frozen.previous.SYSTEM_PROMPT + RULE + APPENDIX


def build_main(model, tools, session, checker, guard=None, replay_prompt=False):
    register_harness_profile('deepseek:'+MODEL, HarnessProfile(
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        excluded_tools={'ls','read_file','write_file','edit_file','delete','glob','grep','execute'}))
    return create_deep_agent(model=model, tools=tools+session.tools(),
        system_prompt=frozen.SYSTEM_PROMPT if replay_prompt else SYSTEM_PROMPT,
        middleware=[guard if guard is not None else frozen.previous.previous.DelegationGuard(),
                    frozen.previous.GrantMiddleware(session), frozen.ConflictMiddleware(checker)],
        subagents=[{'name':'compare','description':frozen.previous.previous.DESCRIPTION,
                    'runnable':frozen.previous.previous.build_compare(model, tools)}])
