"""TL5b: deterministic name extraction from the two authorized sources only."""
import json
import re
import unicodedata
from dataclasses import dataclass, field
from story_agent.report_tl5 import split

RULE = '同一对象有几个名字时（题面里的称呼，以及原文写明是同一对象的其他名字），每个名字都要用 search_quests 和 search_texts 各搜一次；改过名的，一定要用最新的名字搜，后来的经历往往只用新名字。搜索结果被截断时，不能写“查过全部命中”。'


def names(question, answer):
    _, report = split(answer)
    match = re.search(r'(?m)^## 题面前提\s*$', report)
    premise = ''
    if match:
        rest = report[match.end():]
        end = re.search(r'(?m)^## ', rest)
        premise = rest[:end.start()] if end else rest
    found = []
    for m in re.finditer(r'「([^「」]*)」|『([^『』]*)』', question+'\n'+premise):
        value = (m[1] if m[1] is not None else m[2]).strip()
        if (2 <= len(value) <= 6 and not any(c.isspace() or c.isnumeric() or
                unicodedata.category(c).startswith('P') for c in value) and value not in found):
            found.append(value)
    return found


def inspect(question, answer, records):
    searched = {'search_quests': set(), 'search_texts': set()}
    for record in records:
        if record.get('scope') != 'main':
            continue
        for call in record['message'].get('tool_calls', []):
            function = call['function']; name = function['name']
            if name not in searched:
                continue
            args = function['arguments']
            args = json.loads(args) if isinstance(args, str) else args
            query = args.get('query')
            if isinstance(query, str): searched[name].add(query.strip())
    extracted = names(question, answer)
    missing = [{'name': n, 'tools': [t for t in searched if n not in searched[t]]}
               for n in extracted if any(n not in searched[t] for t in searched)]
    return {'names': extracted, 'name_count': len(extracted), 'missing': missing,
            'missing_count': len(missing)}


def feedback(result):
    lines = []
    for item in result['missing']:
        tools = item['tools']
        text = '两者都未用' if len(tools) == 2 else '未用 '+tools[0]
        lines.append('「'+item['name']+'」：'+text)
    return ('交稿前检查：下面这些名字出现在题面或你写的“题面前提”里，但还没用对应工具搜过：\n'
            +'\n'.join(lines)+'\n同一对象有几个名字时，每个都要用 search_quests 和 search_texts 各搜一次；改过名的，后来的经历往往只用新名字。请用这些名字补搜，把搜到的相关原文补进报告，再按原格式重新交稿。')


@dataclass
class NameCheck:
    question: str
    checks: list = field(default_factory=list)
    feedback_count: int = 0

    def examine(self, answer, records):
        result = inspect(self.question, answer, records)
        self.checks.append(result)
        return result

    def feedback_once(self, result):
        if not result['missing'] or self.feedback_count:
            return None
        self.feedback_count = 1
        return feedback(result)

    def metadata(self):
        initial, final = self.checks[0], self.checks[-1]
        return dict(name_count=final['name_count'], names=final['names'],
                    first_missing_count=initial['missing_count'], first_missing=initial['missing'],
                    name_feedback_count=self.feedback_count,
                    resubmitted_missing_count=final['missing_count'], resubmitted_missing=final['missing'])
