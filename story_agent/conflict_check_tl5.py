"""Private conflict reader and item-by-item validation at the return boundary."""
import json
import re
from pathlib import Path
from langchain.agents import create_agent
from langchain_core.tools import StructuredTool, tool
from story_agent.agent_pt0 import REVIEW_TOOLS
from story_agent.community_graph_tl5 import GraphRead, compact, overlaps
from story_agent.strong_agent_sa1 import save

def save_same(path,value):
    if Path(path).exists():
        assert json.loads(Path(path).read_text(encoding='utf-8'))==value
    else: save(path,value)

PROMPT = '你是原文冲突核对者。你会收到另一位研究者交来核对的一组结论。你可以用 read_community_graph 按人名查社区人物图谱，也可以用游戏文本工具读原文和搜索。社区图谱是玩家整理，含猜测，没有出处，只能用来发现“哪条结论可能与别的说法冲突”。你的任务只有一件：找出与游戏原文冲突的结论。每个可能的冲突，都必须自己读到游戏原文，确认原文确实与该结论矛盾，才能列出。不要列出图谱有而结论没写的内容，不要补充新结论，不要提到图谱、社区或任何非游戏文本。只输出 JSON：{"conflicts":[{"conclusion":"从所给结论中原样摘出的句子，最多40字","source":"文件名 L起–止","quote":"该处原文原样摘引，最多60字","note":"矛盾在哪，最多80字"}]}。没有冲突就输出 {"conflicts":[]}。最多 5 条。'

def validate(raw, description, sources, graph_read, game_texts):
    try:
        value = json.loads(raw)
        assert isinstance(value, dict) and isinstance(value['conflicts'], list)
    except (ValueError, TypeError, KeyError, AssertionError):
        return {'structure_failure': True, 'items': [], 'accepted': [], 'submitted': 0}
    checked, accepted = [], []
    for index, item in enumerate(value['conflicts']):
        reasons = []
        fields = ('conclusion', 'source', 'quote', 'note')
        if not isinstance(item, dict) or any(not isinstance(item.get(k), str) or not item[k].strip() for k in fields):
            reasons.append('fields')
        else:
            if index >= 5: reasons.append('more_than_five')
            for k, limit in (('conclusion',40), ('quote',60), ('note',80)):
                if len(item[k]) > limit: reasons.append(k+'_length')
            if compact(item['conclusion']) not in compact(description): reasons.append('conclusion_not_substring')
            refs = sources.references(item['source'])
            loc = refs[0]['resolved'] if len(refs)==1 else None
            if loc is None: reasons.append('source_invalid')
            elif compact(item['quote']) not in compact('\n'.join(loc['lines'])): reasons.append('quote_not_in_lines')
            if any(re.search(r'图谱|社区|GraphLink', item[k], re.I) for k in fields): reasons.append('forbidden_word')
            if overlaps(item['note'], graph_read.notes(), game_texts): reasons.append('notes_leak')
        checked.append({'item': item, 'reasons': reasons, 'accepted': not reasons})
        if not reasons: accepted.append(item)
    return {'structure_failure': False, 'items': checked, 'accepted': accepted, 'submitted': len(checked)}

def addition(accepted):
    if not accepted: return ''
    return '\n\n原文冲突核对（另一位读者逐条对照原文）：\n' + '\n'.join(
        f'{i}. 结论“{v["conclusion"]}”与 {v["source"]} 原文“{v["quote"]}”不符：{v["note"]}'
        for i,v in enumerate(accepted,1))

class ConflictCheck:
    def __init__(self, model, tools, graph, sources, directory):
        self.model, self.tools, self.graph, self.sources = model, tools, graph, sources
        self.directory = Path(directory); self.events = []

    def run(self, description):
        names = self.graph.trigger(description)
        if not names: return ''
        reader = GraphRead(self.graph); reads = []; tool_events = []
        @tool
        def read_community_graph(name: str) -> dict:
            """按人名或别名读社区人物节点；歧义时返回候选，每次核对最多5个不同节点。"""
            return reader.read(name)
        tools = []
        for original in self.tools:
            if original.name not in REVIEW_TOOLS: continue
            def wrap(t):
                def execute(**kwargs):
                    result = t.invoke(kwargs)
                    tool_events.append({'name':t.name, 'args':kwargs, 'result':result})
                    if t.name.startswith('read_') and 'text' in result:
                        # Strip rendered line numbers for whitespace-normalized overlap checks.
                        reads.append(re.sub(r'(?m)^L\d+: ?', '', result['text']))
                    return result
                return execute
            tools.append(StructuredTool.from_function(wrap(original), name=original.name,
                description=original.description, args_schema=original.args_schema, infer_schema=False))
        assert len(tools)==6
        agent = create_agent(self.model, tools=tools+[read_community_graph], system_prompt=PROMPT, name='conflict')
        inputs = description+'\n可查的图谱人物：'+'、'.join(names)
        index = len(self.events)+1
        dest = self.directory/f'{index:03d}'
        save_same(dest/'input.json', {'description':description, 'names':names, 'input':inputs})
        state = agent.invoke({'messages':[{'role':'user','content':inputs}]}, config={'max_concurrency':1})
        raw = state['messages'][-1].text
        result = validate(raw, description, self.sources, reader, reads)
        suffix = addition(result['accepted'])
        record = {'description':description, 'names':names, 'nodes':list(reader.read_nodes.values()),
                  'graph_reads':reader.events, 'game_tools':tool_events, 'raw':raw, 'validation':result, 'addition':suffix}
        self.events.append(record); save_same(dest/'result.json',record)
        return suffix
