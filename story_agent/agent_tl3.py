"""TL3 v0.88.0: append only the two authorized main-system sentences."""
from story_agent import agent_tl2b as baseline
from story_agent.budget import BudgetStopped

VERSION = 'v0.88.0'
PREFIX = 'TL3:' + VERSION + ':'
APPENDIX = '题面里的称谓或说法用字面搜索找不到对应的书，或想知道别的书里有没有相近的说法、异名时，可以用 similar_passages：给出你读到的书名和行号，它会列出其他书里相近的段落，是否真有联系要读原文判断。交稿前，把你打算写成定论、强对应或因果的几条结论交给对照读者核对：写明每条结论、依据的书名与行号，以及同一作者或同一系列的其他篇（可用 list_books 查）；请它查原文有没有推翻或修正这条结论，结论的强度有没有超出原文。'
SYSTEM_PROMPT = baseline.SYSTEM_PROMPT + APPENDIX
DESCRIPTION = baseline.DESCRIPTION
COMPARE_PROMPT = baseline.COMPARE_PROMPT
build_compare = baseline.build_compare
accounted = baseline.accounted


def build_main(model, tools):
    from deepagents import create_deep_agent
    from deepagents.profiles import GeneralPurposeSubagentProfile, HarnessProfile, register_harness_profile
    register_harness_profile('deepseek:' + baseline.MODEL, HarnessProfile(
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        excluded_tools={'ls', 'read_file', 'write_file', 'edit_file', 'delete', 'glob', 'grep', 'execute'},
    ))
    return create_deep_agent(model=model, tools=tools, system_prompt=SYSTEM_PROMPT,
        subagents=[{'name': 'compare', 'description': DESCRIPTION,
                    'runnable': build_compare(model, tools)}])


class AuditedDeepSeek(baseline.AuditedDeepSeek):
    """Keep TL2B wire behavior; isolate TL3's shared-ledger run identifiers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_id = PREFIX + self.topic

    def gate(self):
        summary = self.ledger.summary()
        if summary['limit'] is not None or set(summary['status_counts']) - {'settled'}:
            raise BudgetStopped('Shared ledger must be accounting-only and settled')
        totals = accounted(self.out)
        with self.ledger.connect() as db:
            topics = [p.name for p in self.out.iterdir() if p.is_dir() and (p/'calls').exists()]
            rows = [row for topic in topics for row in db.execute(
                'SELECT status,actual FROM calls WHERE run_id=?', (PREFIX+topic,)).fetchall()]
        if len(rows) != totals['attempts'] or sum(r[1] or 0 for r in rows) != totals['accounted_raw_tokens']:
            raise BudgetStopped('Task ledger and evidence disagree')
        if totals['budget_tokens'] >= self.cap:
            raise BudgetStopped('Uncached input plus output cap reached')
        return totals
