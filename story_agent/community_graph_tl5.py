"""Read-only private community graph; never included in public corpus tools."""
import json
import re
from pathlib import Path

HEADER = '社区整理（GraphLink 人物图谱，2026-05-22 抓取，非游戏文本）：含玩家猜测，没有原文出处。只用来发现可能的冲突；列出的冲突必须有你读到的游戏原文支持。'

def compact(value):
    return re.sub(r'\s+', '', value)

class CommunityGraph:
    def __init__(self, path=None, nodes=None):
        self.nodes = nodes if nodes is not None else json.loads(Path(path).read_text(encoding='utf-8'))

    @staticmethod
    def names(node):
        return list(dict.fromkeys([node['primary_name'], *node.get('aliases', [])]))

    def locate(self, name):
        exact = [n for n in self.nodes if name in self.names(n)]
        return exact or [n for n in self.nodes if any(name in a or a in name for a in self.names(n) if len(a) >= 2)]

    def trigger(self, description):
        return [n['primary_name'] for n in self.nodes if any(len(a) >= 2 and a in description for a in self.names(n))][:8]

class GraphRead:
    def __init__(self, graph):
        self.graph, self.read_nodes, self.events = graph, {}, []

    def read(self, name):
        nodes = self.graph.locate(name)
        if len(nodes) != 1:
            result = {'candidates': [n['primary_name'] for n in nodes], 'error': '未找到唯一人物，请用候选完整姓名。'}
        else:
            n = nodes[0]
            if n['id'] not in self.read_nodes and len(self.read_nodes) >= 5:
                result = {'error': '每次核对最多读取5个不同节点。'}
            else:
                self.read_nodes[n['id']] = n
                value = {k: n.get(k, []) for k in ('primary_name', 'aliases', 'notes')}
                value['outgoing'] = [{k: e[k] for k in ('relation', 'target')} for e in n.get('outgoing', [])]
                value['incoming'] = [{k: e[k] for k in ('relation', 'source')} for e in n.get('incoming', [])]
                body = HEADER + '\n' + json.dumps(value, ensure_ascii=False)
                if len(body) > 6000:
                    body = body[:5996] + '\n未列完'
                result = {'text': body}
        self.events.append({'name': name, 'result': result})
        return result

    def notes(self):
        return [text for n in self.read_nodes.values() for text in n.get('notes', [])]

def overlaps(text, notes, game_texts):
    """Every matching 12-character window must also occur in a read game text."""
    text = compact(text)
    games = [compact(t) for t in game_texts]
    windows = {text[i:i+12] for i in range(max(0, len(text)-11))}
    return sorted(w for w in windows if any(w in compact(n) for n in notes)
                  and not any(w in g for g in games))
