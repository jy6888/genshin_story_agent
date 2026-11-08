"""TL5b v0.94.2: diversify truncated hits without changing tool definitions."""
from collections import Counter
from langchain_core.tools import StructuredTool
from story_agent.texts_tl4b import AgentTools as FrozenTools
from story_agent.agent_tools_tl1 import speaker_for_line

NOTICE='结果已截断：每个文件只列前 3 行命中，文件按命中数排序；要看某文件的全部内容，请读该文件。'


def reshape(result, lines, query, kinds=None, kind=''):
    if not result.get('truncated'):
        return result
    samples={};counts=Counter()
    for name,content in lines.items():
        if kinds is not None and kind and kinds[name]!=kind:continue
        section=''
        for number,line in enumerate(content,1):
            if line.startswith('## '):section=line[3:]
            if query not in line:continue
            counts[name]+=1
            if len(samples.setdefault(name,[]))>=3:continue
            hit=dict(file=name,line=number,text=line,speaker=speaker_for_line(line))
            if kinds is not None:hit.update(kind=kinds[name],section=section)
            samples[name].append(hit)
    matches=[]
    for name in sorted(counts,key=lambda name:(-counts[name],name)):
        matches.extend(samples[name][:80-len(matches)])
        if len(matches)==80:break
    assert sum(counts.values())==result['total_matches']
    return dict(result,matches=matches,notice=NOTICE)


class AgentTools(FrozenTools):
    def __init__(self,*args,use_new=lambda:True,**kwargs):
        super().__init__(*args,**kwargs);self.use_new=use_new

    def search_quests(self,query):
        result=self.quests.search_quests(query)
        return reshape(result,self.quests.lines,query) if self.use_new() else result

    def search_texts(self,query,kind=''):
        result=self.texts.search_texts(query,kind)
        return reshape(result,self.texts.lines,query,self.texts.kinds,kind) if self.use_new() else result

    def tools(self):
        result=[]
        for tool in super().tools():
            if tool.name not in ('search_quests','search_texts'):
                result.append(tool);continue
            function=self.search_quests if tool.name=='search_quests' else self.search_texts
            result.append(StructuredTool.from_function(function,name=tool.name,
                description=tool.description,args_schema=tool.args_schema,infer_schema=False))
        return result
