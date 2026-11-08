"""D-022: top-level guesses need labels; complete filenames may abut prose."""
import json
import re
from pathlib import Path

from story_agent import report_default_v952 as frozen
from story_agent.report_tl5 import LINE, split, sections
from story_agent.report_tl5_v934 import Sources as FrozenSources

VERSION = 'v0.95.3'
FIELDS = ('抓手', '为什么', '如何检验', '依据', '>')
NameCheck = frozen.NameCheck
format_feedback = frozen.format_feedback

FULL = re.compile(r'((?:agd_[\w]+/)?\d{2,}_[^\s`，。；：、（）()\[\]【】<>]+?\.txt)\s*[`：:，,、]?\s*[（(]?\s*' + LINE)


class Sources(FrozenSources):
    def references(self, text):
        refs = super().references(text)
        complete = {}
        for match in FULL.finditer(text):
            resolved = self.resolve(match[1], int(match[2]), int(match[3] or match[2]))
            if resolved is not None:
                complete[match.end()] = (match[0], resolved)
        for ref in refs:
            if ref['resolved'] is not None:
                continue
            candidates = {}
            for occurrence in re.finditer(re.escape(ref['raw']), text):
                if occurrence.end() not in complete:
                    continue
                raw, resolved = complete[occurrence.end()]
                key = (resolved['file'], resolved['start'], resolved['end'])
                candidates[key] = (raw, resolved)
            if len(candidates) == 1:
                raw, resolved = next(iter(candidates.values()))
                ref.update(raw=raw, resolved=resolved)
        return refs


def check(answer, sources):
    errors = [e for e in frozen.check(answer, sources)
              if not e.startswith(('结论节：', '猜测节：'))]
    for name, body in sections(split(answer)[1]).items():
        if name not in ('结论', '猜测'):
            continue
        for paragraph in re.split(r'\n\s*\n', body):
            first = True
            for raw in paragraph.splitlines():
                line = raw.strip()
                if not line or line in frozen.EMPTY or line.startswith('### '):
                    continue
                bullet = bool(re.match(r'^(?:[-*]|\d+[.、])\s+', line))
                line = re.sub(r'^(?:[-*]|\d+[.、])\s+', '', line)
                if line.startswith(FIELDS):
                    continue
                if not bullet and not first:
                    continue
                first = False
                allowed = r'^【(?:明说(?:·人物说法)?|推导)】' if name == '结论' else r'^【猜测】'
                if not re.match(allowed, line):
                    wording = ('每条猜测以【猜测】开头；抓手、为什么前两级给不出、如何检验写在该条下面；没有猜测就写“无”。'
                        if name == '猜测' else '每条结论以【明说】【明说·人物说法】或【推导】开头；依据、引文等字段写在该条下面；没有结论就写“无”。')
                    errors.append(f'{name}节：{line[:55]}；{wording}')
    return errors


def deliver(answer, directory, sources, metadata, feedback_count):
    result = frozen.deliver(answer, directory, sources, metadata, feedback_count)
    result['errors'] = check(answer, sources)
    result['metadata'].update(final_invalid_count=len(result['errors']), format_errors=result['errors'],
                              postprocessor_version=VERSION)
    path = Path(directory)/'report.md'
    prefix = path.read_text(encoding='utf-8').rsplit('## 附录 B：运行信息', 1)[0]
    path.write_text(prefix+'## 附录 B：运行信息\n\n```json\n'+
        json.dumps(result['metadata'], ensure_ascii=False, indent=2)+'\n```\n', encoding='utf-8')
    return result
