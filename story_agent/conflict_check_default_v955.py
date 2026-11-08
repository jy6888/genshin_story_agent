"""D-20261108-001: same-file multi-location citations and one format reply."""
import json
import re
from unittest.mock import patch

from story_agent import conflict_check_tl5_v931 as previous
from story_agent.community_graph_tl5 import compact, overlaps
from story_agent.report_tl5 import LINE

PROMPT=previous.PROMPT
VERSION='v0.95.5'
FIELDS=('conclusion','source','quote','note')
LIMITS={'conclusion':40,'quote':60,'note':80}
DIRECT={'forbidden_word','notes_leak'}


def locations(text,sources,events=()):
    """Resolve every location to one file; never silently ignore suffix text."""
    if not isinstance(text,str):return None
    matches=list(re.finditer(LINE,text))
    if not matches:return None
    name=text[:matches[0].start()].strip().strip('《》`* ')
    if not name or text[matches[-1].end():].strip():return None
    for left,right in zip(matches,matches[1:]):
        if not re.fullmatch(r'\s*[/、，;]\s*',text[left.end():right.start()]):return None
    candidates=list(dict.fromkeys(sources.aliases.get(name,[])))
    read_files={e['result'].get('file') for e in events
                if e['name'].startswith('read_') and isinstance(e.get('result'),dict) and 'text' in e['result']}
    if len(candidates)!=1:
        candidates=[n for n in candidates if n in read_files]
    if len(candidates)!=1:return None
    file=candidates[0];resolved=[]
    for match in matches:
        value=sources.resolve(file,int(match[1]),int(match[2] or match[1]))
        if value is None:return None
        resolved.append(value)
    return dict(file=file,ranges=[[v['start'],v['end']] for v in resolved],
                lines=[line for v in resolved for line in v['lines']])


class MultiSources:
    def __init__(self,sources,events):self.sources,self.events,self.resolutions=sources,events,[]
    def references(self,text):
        value=locations(text,self.sources,self.events)
        if value is not None:self.resolutions.append(dict(source=text,**value))
        return [dict(raw=text,resolved=value)]


def validate(raw,description,sources,reader,reads,events=()):
    value,parsing=previous.parse(raw);scoped=MultiSources(sources,events)
    result=previous.frozen.validate(json.dumps(value,ensure_ascii=False) if value is not None else '',
        description,scoped,reader,reads)
    # A missing field must not mask a direct privacy rejection.
    for record in result['items']:
        item=record['item']
        if isinstance(item,dict):
            if any(isinstance(item.get(k),str) and re.search(r'图谱|社区|GraphLink',item[k],re.I) for k in FIELDS):
                if 'forbidden_word' not in record['reasons']:record['reasons'].append('forbidden_word')
            if isinstance(item.get('note'),str) and overlaps(item['note'],reader.notes(),reads):
                if 'notes_leak' not in record['reasons']:record['reasons'].append('notes_leak')
        record['accepted']=not record['reasons']
    result['accepted']=[r['item'] for r in result['items'] if r['accepted']]
    result.update(parsing=parsing,source_resolutions=scoped.resolutions)
    return result


def candidates(item,sources,events):
    source=item.get('source','') if isinstance(item,dict) else ''
    found=locations(source,sources,events)
    if found:
        return '可用完整标识：'+found['file']+' '+ '、'.join(f'L{a}–{b}' if a!=b else f'L{a}' for a,b in found['ranges'])+\
            '；这些行的原文：'+ '\n'.join(found['lines'])[:600]
    match=re.search(LINE,source) if isinstance(source,str) else None
    name=source[:match.start()].strip().strip('《》`* ') if match else str(source).strip()
    names=list(dict.fromkeys(sources.aliases.get(name,[])))
    if not names:
        names=list(dict.fromkeys(e['result'].get('file') for e in events if e['name'].startswith('read_')
            and isinstance(e.get('result'),dict) and e['result'].get('file') in sources.lines))
    if not names:names=list(sources.lines)[:5]
    return '可用完整标识与行号边界：'+'；'.join(f'{n} L1–{len(sources.lines[n])}' for n in names[:6])


def feedback(result,description,sources,events=()):
    problems=[]
    for index,record in enumerate(result['items'],1):
        reasons=record['reasons'];item=record['item']
        if DIRECT.intersection(reasons):continue
        for reason in reasons:
            if reason=='fields':message='允许字段conclusion、source、quote、note，四项均须为非空字符串。'
            elif reason=='more_than_five':message='conflicts最多5条；删除多出的条目，保留要核对的至多5条。'
            elif reason.endswith('_length') and reason[:-7] in LIMITS:
                field=reason[:-7];message=f'{field}最多{LIMITS[field]}字，缩短该字段后重交。'
            elif reason=='conclusion_not_substring':
                message='conclusion须从本轮输入结论逐字摘取，最多40字；输入开头的可用片段：'+description[:40]
            elif reason=='source_invalid':
                message='source只写同一文件的完整标识与存在的行号；允许“文件.txt L130/L336”或“文件.txt L130–132、L335–336”，不得跨文件。'+candidates(item,sources,events)
            elif reason=='quote_not_in_lines':
                message='quote最多60字，须从source各处行号合并后的原文逐字摘取（仅忽略空白）。'+candidates(item,sources,events)
            else:continue
            problems.append(f'第{index}条 {reason}：{message}')
    if not problems:return None
    return ('只修正以下格式问题，不新增冲突、不改变原文判断；完整重交同一JSON结构'+
        '{"conflicts":[{"conclusion":"…","source":"…","quote":"…","note":"…"}]}。本轮只反馈一次，重交仍不合格的条目会被剔除。\n'+'\n'.join(problems))


def review_once(raw,description,sources,reader,reads,events,resubmit):
    initial=validate(raw,description,sources,reader,reads,events)
    message=feedback(initial,description,sources,events)
    if message is None:return dict(initial,format_feedback_count=0)
    final_raw=resubmit(message)
    final=validate(final_raw,description,sources,reader,reads,events)
    blocked={compact(r['item']['conclusion']):DIRECT.intersection(r['reasons']) for r in initial['items']
        if isinstance(r['item'],dict) and isinstance(r['item'].get('conclusion'),str) and DIRECT.intersection(r['reasons'])}
    # An unrelated format retry cannot revive an already directly deleted item.
    for record in final['items']:
        item=record['item'];conclusion=item.get('conclusion') if isinstance(item,dict) else None
        key=compact(conclusion) if isinstance(conclusion,str) else ''
        for reason in blocked.get(key,()):
            if reason not in record['reasons']:record['reasons'].append(reason)
        record['accepted']=not record['reasons']
    final['accepted']=[r['item'] for r in final['items'] if r['accepted']]
    return dict(final,format_feedback_count=1,format_feedback=message,initial_raw=raw,
        initial_validation=initial,final_raw=final_raw)


class ConflictCheck(previous.ConflictCheck):
    def run(self,description):
        # Reuse the frozen tool definitions, prompt and exact first invocation.
        factory=previous.create_agent;holder={}
        def capture_factory(*args,**kwargs):
            agent=factory(*args,**kwargs);holder['agent']=agent
            class Captured:
                def invoke(self,*values,**options):
                    holder['state']=agent.invoke(*values,**options)
                    return holder['state']
            return Captured()
        def check(raw,desc,sources,reader,reads,events=()):
            def resubmit(message):
                state=holder['agent'].invoke({'messages':holder['state']['messages']+[{'role':'user','content':message}]},
                                            config={'max_concurrency':1})
                holder['state']=state
                return state['messages'][-1].text
            return review_once(raw,desc,sources,reader,reads,events,resubmit)
        with patch.object(previous,'create_agent',capture_factory),patch.object(previous,'validate',check):
            return super().run(description)
