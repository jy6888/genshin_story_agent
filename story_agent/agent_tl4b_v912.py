"""TL4b §10: append the one authorized full-text sentence, nothing else."""
from langchain.agents import create_agent
from story_agent import agent_tl4b as previous
from story_agent.budget import BudgetStopped

VERSION = 'v0.91.2'
PREFIX = 'TL4b:' + VERSION + ':'
APPENDIX = '核对时，角色故事、角色语音和三百行以内的书请读全文，不只读给定的行号。'
COMPARE_PROMPT = previous.COMPARE_PROMPT + APPENDIX
SYSTEM_PROMPT = previous.SYSTEM_PROMPT
DESCRIPTION = previous.DESCRIPTION
MODEL = previous.MODEL
CHILD_TOOLS = previous.CHILD_TOOLS
MAIN_TOOLS = previous.MAIN_TOOLS
DelegationGuard = previous.DelegationGuard
accounted = previous.accounted


def build_compare(model, tools):
    selected = [t for t in tools if t.name in CHILD_TOOLS]
    if {t.name for t in selected} != CHILD_TOOLS or len(selected) != 6:
        raise BudgetStopped('Compare requires six authorized read/search tools')
    return create_agent(model, tools=selected, system_prompt=COMPARE_PROMPT, name='compare')


def build_main(model, tools, guard=None):
    from deepagents import create_deep_agent
    from deepagents.profiles import GeneralPurposeSubagentProfile, HarnessProfile, register_harness_profile
    register_harness_profile('deepseek:' + MODEL, HarnessProfile(
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        excluded_tools={'ls', 'read_file', 'write_file', 'edit_file', 'delete', 'glob', 'grep', 'execute'}))
    return create_deep_agent(model=model, tools=tools, system_prompt=SYSTEM_PROMPT,
        middleware=[guard if guard is not None else DelegationGuard()],
        subagents=[{'name': 'compare', 'description': DESCRIPTION,
                    'runnable': build_compare(model, tools)}])


class AuditedDeepSeek(previous.AuditedDeepSeek):
    """Identical transport; independent version namespace and evidence budget."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_id = PREFIX + self.out.name + ':' + self.topic

    def gate(self):
        self.monitor.check_stop()
        summary = self.ledger.summary()
        if summary['limit'] is not None or set(summary['status_counts']) - {'settled'}:
            raise BudgetStopped('Shared ledger must be accounting-only and settled')
        totals = accounted(self.out)
        with self.ledger.connect() as db:
            topics = [p.name for p in self.out.iterdir() if p.is_dir() and (p/'calls').exists()]
            rows = [r for topic in topics for r in db.execute(
                'SELECT status,actual FROM calls WHERE run_id=?',
                (PREFIX+self.out.name+':'+topic,)).fetchall()]
        if len(rows) != totals['attempts'] or sum(r[1] or 0 for r in rows) != totals['accounted_raw_tokens']:
            raise BudgetStopped('Task ledger and evidence disagree')
        if totals['budget_tokens'] > self.cap:
            raise BudgetStopped('Uncached input plus output cap exceeded')
        return totals
