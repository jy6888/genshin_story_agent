"""TL5 source resolver, one-feedback format checks and deterministic appendices."""
import json
import re
from pathlib import Path

HEADINGS = ['问题', '简答', '题面前提', '结论', '猜测', '查过但没找到', '未解决与下一步']
SEPARATOR = '=====报告====='
APPENDIX = '交稿时依次输出两部分，中间用单独一行“=====报告=====”分隔。第一部分是给玩家的对话总结，300–600字：第一句直接回答问题，答不了就写“原文没有给出答案”，再说最接近的线索；然后列3–6个要点，每个以【明说】【推导】【猜测】之一开头；用了外部资料的，单列一句，以“原型线索（外部资料）：”开头；不写报告里没有的结论；用口语，不提工具名。第二部分是报告，按顺序使用这七个二级标题：## 问题、## 简答、## 题面前提、## 结论、## 猜测、## 查过但没找到、## 未解决与下一步；某节没有内容就写“无”。“结论”按小问题分 ### 小节，每条以【明说】或【推导】开头，人物台词写作【明说·人物说法】；【明说】附出处和不超过40字的原文摘引；【推导】写成“依据：甲（出处）＋乙（出处）→ 结论”。“猜测”每条写四项：猜测内容、抓手（出处）、为什么前两级给不出、如何检验。“查过但没找到”每条写查过的范围（书、任务、文本类型、搜索词）。出处写“文件名 L起–止”，例如“1076_至冬国通史 L19–29”。不要写附录，程序会补。'
LINE = r'L\s*(\d+)(?:\s*[-–—~～至到]\s*L?\s*(\d+))?'
REF = re.compile(r'(《[^》\n]+》|(?:agd_[\w]+/)?\d{2,}_[^\s`，。；：、（）()\[\]【】<>]+|[\w\u3400-\u9fff/·「」\-]+\.txt)\s*[`：:，,、]?\s*[（(]?\s*' + LINE)

class Sources:
    def __init__(self, books):
        self.lines = {**books.lines, **books.quests.lines, **books.texts.lines}
        self.aliases = {}
        for name in self.lines:
            stem = Path(name).stem
            for alias in {name, Path(name).name, stem, re.sub(r'^\d+_', '', stem)}:
                self.aliases.setdefault(alias, []).append(name)
        for collection in (books.aliases, books.quests.aliases, books.texts.aliases):
            for alias, names in collection.items():
                self.aliases.setdefault(alias, []).extend(names)

    def resolve(self, file, start, end):
        file = file.strip('《》`* ')
        names = list(dict.fromkeys(self.aliases.get(file, [])))
        if len(names) != 1:
            return None
        name = names[0]
        if not (1 <= start <= end <= len(self.lines[name])):
            return None
        return {'file': name, 'start': start, 'end': end, 'lines': self.lines[name][start-1:end]}

    def references(self, text):
        result, covered = [], []
        for m in REF.finditer(text):
            start, end = int(m[2]), int(m[3] or m[2])
            loc = self.resolve(m[1], start, end)
            # Historical quests often prefix a potentially ambiguous title with its ID.
            prefix = text[max(0,m.start()-12):m.start()]
            number = re.search(r'(\d{3,})\s*$',prefix)
            if loc is None and number:
                names=[n for n in self.lines if Path(n).name.startswith(number[1]+'_')]
                if len(names)==1: loc=self.resolve(names[0],start,end)
            result.append({'raw': m[0], 'resolved': loc})
            covered.append((m.start(),m.end()))
        # Never silently omit a line-number citation whose filename is abbreviated.
        for m in re.finditer(LINE,text):
            if any(a<=m.start()<b for a,b in covered):continue
            start,end=int(m[1]),int(m[2] or m[1])
            paragraph=text[text.rfind('\n\n',0,m.start())+2:m.start()]
            explicit=list(REF.finditer(paragraph))
            loc=None
            if explicit:
                loc=self.resolve(explicit[-1][1],start,end)
            raw=text[max(text.rfind('\n',0,m.start())+1,m.start()-55):m.end()]
            result.append({'raw':raw,'resolved':loc})
        return result

def split(answer):
    parts = re.split(r'(?m)^\s*=====报告=====\s*$', answer, maxsplit=1)
    return (parts[0].strip(), parts[1].strip()) if len(parts) == 2 else ('', answer.strip())

def sections(report):
    matches = list(re.finditer(r'(?m)^##\s+([^\n]+)\s*$', report))
    return {m[1].strip(): report[m.end():matches[i+1].start() if i+1 < len(matches) else len(report)].strip()
            for i, m in enumerate(matches)}

def check(answer, sources):
    errors = []
    if not re.search(r'(?m)^=====报告=====\s*$', answer):
        errors.append('分隔行缺失；允许写法：单独一行 =====报告=====。')
    summary, report = split(answer)
    found = re.findall(r'(?m)^##\s+([^\n]+)', report)
    if found != HEADINGS:
        errors.append('报告二级标题或顺序不符；允许写法依次为：' + '、'.join('## '+h for h in HEADINGS))
    for name, body in sections(report).items():
        if name not in ('结论', '猜测'):
            continue
        for paragraph in re.split(r'\n\s*\n', body):
            paragraph = paragraph.strip()
            if not paragraph or paragraph == '无':
                continue
            if paragraph.startswith('### '):
                # A subsection can immediately precede its first conclusion.
                paragraph = re.sub(r'^### [^\n]+\n?', '', paragraph).strip()
                if not paragraph:
                    continue
            for line in paragraph.splitlines():
                line = re.sub(r'^\s*(?:[-*]|\d+[.、])\s*', '', line).strip()
                if not line or line.startswith(('依据：', '抓手', '为什么', '如何', '猜测内容', '>')):
                    continue
                allowed = r'^【(?:明说(?:·人物说法)?|推导)】' if name == '结论' else r'^【猜测】'
                if not re.match(allowed, line):
                    errors.append(f'{name}节：{line[:55]}；每条允许以' + ('【明说】【明说·人物说法】或【推导】开头。' if name == '结论' else '【猜测】开头。'))
                break
    for ref in sources.references(report):
        if ref['resolved'] is None:
            errors.append('出处无法定位：' + ref['raw'] + '；允许写法：唯一文件名 L起–止（文件与行号均须存在）。')
    return errors

def feedback(errors):
    return '报告格式未通过检查，请只按下列问题修改后重新交稿，内容不要删减：\n' + '\n'.join(f'{i}. {v}' for i,v in enumerate(errors,1))

def appendix_a(report, sources):
    body = sections(report)
    refs = sources.references('\n'.join(body.get(h, '') for h in ('题面前提', '结论', '猜测')))
    seen, output = set(), ['## 附录 A：引用原文']
    unique = []
    for ref in refs:
        if ref['raw'] in seen: continue
        seen.add(ref['raw']); unique.append(ref)
        loc = ref['resolved']
        if loc is None:
            output.append(f"\n{len(unique)}. {ref['raw']}：出处无法定位")
        else:
            output.append(f"\n{len(unique)}. {loc['file']} L{loc['start']}–{loc['end']}\n")
            output.extend(f"> L{loc['start']+i}: {line}" for i,line in enumerate(loc['lines'][:5]))
            if len(loc['lines']) > 5: output.append('> 超过5行，已截断。')
    return '\n'.join(output), unique

def deliver(answer, directory, sources, metadata, feedback_count=0):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    errors = check(answer, sources)
    summary, report = split(answer)
    parts = sections(report)
    # Preserve all content; insert empty sections and canonical order on final failure.
    if re.findall(r'(?m)^##\s+([^\n]+)', report) != HEADINGS:
        preamble = re.split(r'(?m)^## ', report, maxsplit=1)[0].strip()
        report = ('\n\n'.join('## '+h+'\n\n'+parts.get(h, '无') for h in HEADINGS)
                  + ('\n\n'+preamble if preamble else '')
                  + ''.join('\n\n## '+h+'\n\n'+v for h,v in parts.items() if h not in HEADINGS))
    app, refs = appendix_a(report, sources)
    metadata = dict(metadata, format_feedback_count=feedback_count, final_invalid_count=len(errors),
                    unresolved_sources=sum(r['resolved'] is None for r in refs), format_errors=errors)
    appendix_b = '## 附录 B：运行信息\n\n```json\n'+json.dumps(metadata, ensure_ascii=False, indent=2)+'\n```'
    (directory/'summary.md').write_text(summary+'\n\n完整报告：'+str((directory/'report.md').resolve())+'\n', encoding='utf-8')
    (directory/'report.md').write_text(report+'\n\n'+app+'\n\n'+appendix_b+'\n', encoding='utf-8')
    return {'errors': errors, 'references': refs, 'metadata': metadata}
