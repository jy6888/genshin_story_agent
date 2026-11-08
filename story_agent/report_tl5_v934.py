"""Offline-only final validator repair: every list item, same-line explicit source."""
import json
import re
from pathlib import Path
from story_agent import report_tl5 as original
from story_agent.report_tl5_v932 import Sources as Previous

BASE_CHECK=original.check

class Sources(Previous):
    def references(self,text):
        refs=super().references(text)
        for ref,match in zip(refs,re.finditer(original.LINE,text)):
            if ref['resolved'] is not None:continue
            line=text[text.rfind('\n',0,match.start())+1:match.start()]
            names=re.findall(r'agd_[\w]+/[^\s`（）()，；：]+\.txt',line)
            names=list(dict.fromkeys(n for n in names if n in self.lines))
            if len(names)==1:
                start,end=int(match[1]),int(match[2] or match[1])
                resolved=self.resolve(names[0],start,end)
                if resolved:
                    ref.update(raw=names[0]+' '+match[0],resolved=resolved)
        return refs

def check(answer,sources):
    errors=[e for e in BASE_CHECK(answer,sources) if not e.startswith(('结论节：','猜测节：'))]
    _,report=original.split(answer)
    for name,body in original.sections(report).items():
        if name not in ('结论','猜测'):continue
        for paragraph in re.split(r'\n\s*\n',body):
            first=True
            for raw in paragraph.splitlines():
                line=raw.strip()
                if not line or line=='无' or line.startswith('### '):continue
                bullet=bool(re.match(r'^(?:[-*]|\d+[.、])\s+',line))
                if not bullet and not first:continue
                first=False
                line=re.sub(r'^(?:[-*]|\d+[.、])\s+','',line)
                allowed=r'^【(?:明说(?:·人物说法)?|推导)】' if name=='结论' else r'^【猜测】'
                if not re.match(allowed,line):
                    errors.append(f'{name}节：{line[:55]}；每条允许以'+('【明说】【明说·人物说法】或【推导】开头。' if name=='结论' else '【猜测】开头。'))
    return errors

def deliver(answer,directory,sources,metadata,feedback_count):
    """Keep model text intact; reissue only derived files with corrected diagnostics."""
    result=original.deliver(answer,directory,sources,metadata,feedback_count)
    summary,_=original.split(answer);errors=check(answer,sources)
    result['errors']=errors
    result['metadata'].update(final_invalid_count=len(errors),format_errors=errors,
        postprocessor_version='v0.93.4',model_text_unchanged=True,
        summary_missing=not summary.strip(),summary_characters=len(re.sub(r'\s+','',summary)))
    path=Path(directory)/'report.md';text=path.read_text(encoding='utf-8')
    text=text.rsplit('## 附录 B：运行信息',1)[0]+'## 附录 B：运行信息\n\n```json\n'+json.dumps(result['metadata'],ensure_ascii=False,indent=2)+'\n```\n'
    path.write_text(text,encoding='utf-8')
    return result
