"""D-20261107-013: tolerant JSON envelope, unchanged private prompt and rules."""
import json
import re
from pathlib import Path
from langchain.agents import create_agent
from langchain_core.tools import StructuredTool, tool
from story_agent import conflict_check_tl5 as frozen
from story_agent.agent_pt0 import REVIEW_TOOLS
from story_agent.community_graph_tl5 import GraphRead

PROMPT=frozen.PROMPT

def parse(raw):
    decoder=json.JSONDecoder()
    try:
        value=json.loads(raw)
        return value,{'mode':'whole','with_explanation':False,'discarded_text':''}
    except (ValueError,TypeError):pass
    blocks=list(re.finditer(r'```json\s*\n?(.*?)```',raw,re.S|re.I))
    if blocks:
        selected=blocks[-1]
        try:value=json.loads(selected[1])
        except ValueError:return None,{'mode':'invalid_last_block','with_explanation':True,'discarded_text':raw}
        rest=(raw[:selected.start()]+raw[selected.end():]).strip()
        return value,{'mode':'last_json_block','with_explanation':bool(rest),'discarded_text':rest}
    candidates=[]; position=0
    while position<len(raw):
        at=raw.find('{',position)
        if at<0:break
        try:value,length=decoder.raw_decode(raw[at:])
        except ValueError:position=at+1;continue
        end=at+length
        if isinstance(value,dict) and 'conflicts' in value:candidates.append((value,at,end))
        position=end
    if len(candidates)!=1:
        return None,{'mode':'ambiguous' if candidates else 'none','candidate_count':len(candidates),
                     'with_explanation':bool(raw.strip()),'discarded_text':raw}
    value,start,end=candidates[0];rest=(raw[:start]+raw[end:]).strip()
    return value,{'mode':'embedded_unique','with_explanation':bool(rest),'discarded_text':rest}

class ReadSources:
    """Resolve a repeated basename only if exactly one matching file was actually read."""
    def __init__(self,sources,events):
        self.sources=sources
        self.read_files={e['result'].get('file') for e in events if e['name'].startswith('read_') and 'text' in e['result']}
        self.resolutions=[]

    def references(self,text):
        refs=self.sources.references(text)
        for ref in refs:
            if ref['resolved'] is not None:continue
            match=frozen.re.search(__import__('story_agent.report_tl5',fromlist=['REF']).REF,text)
            if not match:continue
            name=match[1].strip('《》`* ');start=int(match[2]);end=int(match[3] or match[2])
            candidates=list(dict.fromkeys(self.sources.aliases.get(name,[])))
            read=[n for n in candidates if n in self.read_files]
            if len(read)==1:
                ref['resolved']=self.sources.resolve(read[0],start,end)
                if ref['resolved'] is not None:self.resolutions.append({'source':ref['raw'],'resolved_file':read[0],
                    'basis':'同名标识只在本次实际读取的文件中命中一个；行号与逐字摘引另行校验。'})
        return refs

def validate(raw,description,sources,reader,reads,events=()):
    value,parsing=parse(raw)
    scoped=ReadSources(sources,events)
    strict=json.dumps(value,ensure_ascii=False) if value is not None else ''
    result=frozen.validate(strict,description,scoped,reader,reads)
    result['parsing']=parsing;result['source_resolutions']=scoped.resolutions
    return result

class ConflictCheck(frozen.ConflictCheck):
    def run(self,description):
        names=self.graph.trigger(description)
        if not names:return ''
        reader=GraphRead(self.graph);reads=[];events=[]
        @tool
        def read_community_graph(name:str)->dict:
            """按人名或别名读社区人物节点；歧义时返回候选，每次核对最多5个不同节点。"""
            return reader.read(name)
        tools=[]
        for original in self.tools:
            if original.name not in REVIEW_TOOLS:continue
            def wrap(t):
                def execute(**kwargs):
                    result=t.invoke(kwargs);events.append({'name':t.name,'args':kwargs,'result':result})
                    if t.name.startswith('read_') and 'text' in result:
                        reads.append(re.sub(r'(?m)^L\d+: ?', '',result['text']))
                    return result
                return execute
            tools.append(StructuredTool.from_function(wrap(original),name=original.name,description=original.description,
                         args_schema=original.args_schema,infer_schema=False))
        assert len(tools)==6
        agent=create_agent(self.model,tools=tools+[read_community_graph],system_prompt=PROMPT,name='conflict')
        inputs=description+'\n可查的图谱人物：'+'、'.join(names)
        dest=self.directory/f'{len(self.events)+1:03d}'
        frozen.save_same(dest/'input.json',dict(description=description,names=names,input=inputs))
        state=agent.invoke({'messages':[{'role':'user','content':inputs}]},config={'max_concurrency':1})
        raw=state['messages'][-1].text
        result=validate(raw,description,self.sources,reader,reads,events)
        suffix=frozen.addition(result['accepted'])
        record=dict(description=description,names=names,nodes=list(reader.read_nodes.values()),graph_reads=reader.events,
                    game_tools=events,raw=raw,validation=result,addition=suffix)
        self.events.append(record);frozen.save_same(dest/'result.json',record)
        return suffix
