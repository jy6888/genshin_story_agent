"""PT0 unchanged main/compare tool surfaces; private post-compare check."""
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from dataclasses import replace
from story_agent import agent_pt0 as previous
from story_agent.report_tl5 import APPENDIX

VERSION='v0.93.0'
MODEL=previous.MODEL
SYSTEM_PROMPT=previous.SYSTEM_PROMPT+APPENDIX

class ConflictMiddleware(AgentMiddleware):
    def __init__(self, checker): self.checker=checker

    def append(self, request, result):
        call=request.tool_call
        if call['name']=='task' and call.get('args',{}).get('subagent_type')=='compare':
            suffix=self.checker.run(call['args']['description'])
            if suffix:
                if isinstance(result, Command):
                    update=dict(result.update)
                    messages=list(update.get('messages',[]))
                    for i in range(len(messages)-1,-1,-1):
                        if isinstance(messages[i],ToolMessage) and messages[i].tool_call_id==call['id']:
                            messages[i]=messages[i].model_copy(update={'content':messages[i].content+suffix})
                            break
                    else:
                        raise RuntimeError('Compare command missing its ToolMessage')
                    update['messages']=messages
                    result=replace(result,update=update)
                else:
                    result=result.model_copy(update={'content':result.content+suffix})
        return result

    def wrap_tool_call(self, request, handler):
        return self.append(request,handler(request))

    async def awrap_tool_call(self, request, handler):
        return self.append(request,await handler(request))

def build_main(model, tools, session, checker, guard=None):
    from deepagents import create_deep_agent
    from deepagents.profiles import GeneralPurposeSubagentProfile, HarnessProfile, register_harness_profile
    register_harness_profile('deepseek:'+MODEL,HarnessProfile(
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        excluded_tools={'ls','read_file','write_file','edit_file','delete','glob','grep','execute'}))
    return create_deep_agent(model=model,tools=tools+session.tools(),system_prompt=SYSTEM_PROMPT,
        middleware=[guard if guard is not None else previous.previous.DelegationGuard(),
                    previous.GrantMiddleware(session),ConflictMiddleware(checker)],
        subagents=[{'name':'compare','description':previous.previous.DESCRIPTION,
                    'runnable':previous.previous.build_compare(model,tools)}])
