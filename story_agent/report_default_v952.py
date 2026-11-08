"""D-20261107-021: field-aware diagnostics, unchanged frozen report modules."""
import json
import re
from pathlib import Path

from story_agent import report_tl5b_v942 as frozen
from story_agent.report_tl5 import split, sections

VERSION = 'v0.95.2'
EMPTY = {'无', '无。', '暂无', '没有'}
FIELDS = ('抓手', '为什么', '如何检验', '猜测内容', '依据', '>')
NameCheck = frozen.NameCheck
HEADER = '报告格式未通过检查，请只按下列问题修改后重新交稿，内容不要删减；只改格式，不新增结论。某节没有内容就写“无”。确需新增结论，先交对照读者核对。'


def check(answer, sources):
    # Keep source, heading and summary checks; replace only entry recognition.
    errors = [e for e in frozen.check(answer, sources)
              if not e.startswith(('结论节：', '猜测节：'))]
    _, report = split(answer)
    for name, body in sections(report).items():
        if name not in ('结论', '猜测'):
            continue
        for paragraph in re.split(r'\n\s*\n', body):
            first = True
            for raw in paragraph.splitlines():
                line = raw.strip()
                if not line or line in EMPTY or line.startswith('### '):
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
                        if name == '猜测' else
                        '每条结论以【明说】【明说·人物说法】或【推导】开头；依据、引文等字段写在该条下面；没有结论就写“无”。')
                    errors.append(f'{name}节：{line[:55]}；{wording}')
    return errors


def format_feedback(errors):
    return HEADER + '\n' + '\n'.join(f'{i}. {error}' for i, error in enumerate(errors, 1))


def deliver(answer, directory, sources, metadata, feedback_count):
    result = frozen.deliver(answer, directory, sources, metadata, feedback_count)
    result['errors'] = check(answer, sources)
    result['metadata'].update(final_invalid_count=len(result['errors']),
        format_errors=result['errors'], postprocessor_version=VERSION)
    path = Path(directory)/'report.md'
    prefix = path.read_text(encoding='utf-8').rsplit('## 附录 B：运行信息', 1)[0]
    path.write_text(prefix+'## 附录 B：运行信息\n\n```json\n'+
        json.dumps(result['metadata'], ensure_ascii=False, indent=2)+'\n```\n', encoding='utf-8')
    return result
