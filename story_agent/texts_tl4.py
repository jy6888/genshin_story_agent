"""TL4: immutable UTF-8 snapshot of the sixteen authorized game-text kinds."""
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import re

from langchain_core.tools import tool
from story_agent.agent_tools_tl1 import speaker_for_line, unread_notice
from story_agent.similar_passages_tl2a_v0862 import AgentTools as FrozenTools

KINDS = ('character_story', 'voiceline', 'hangout', 'talk_group', 'talk',
         'material_type', 'weapon', 'artifact_set', 'anecdote', 'subtitle',
         'readable', 'achievement', 'creature', 'costume', 'wings', 'blossom')


class Texts:
    def __init__(self, root):
        root = Path(root).resolve()
        self.lines, self.kinds, self.hashes = {}, {}, {}
        self.aliases = defaultdict(list)
        for kind in KINDS:
            corpus = root / '.runtime/istaroth-checkpoint/text' / ('agd_' + kind)
            for path in (corpus, *corpus.parents):
                if path == root:
                    break
                if path.is_symlink() or path.is_junction():
                    raise ValueError('Text corpus directory link rejected')
            if not corpus.is_dir():
                raise ValueError('Fixed text corpus missing: ' + kind)
            paths = sorted(corpus.glob('*.txt'))
            if not paths:
                raise ValueError('Fixed text corpus empty: ' + kind)
            for path in paths:
                if (path.is_symlink() or path.is_junction() or not path.is_file()
                        or path.resolve().parent != corpus):
                    raise ValueError('Text outside fixed corpus')
                # Identical basenames occur in story/voiceline. Fixed relative
                # identifiers disambiguate them without accepting filesystem paths.
                name = corpus.name + '/' + path.name
                raw = path.read_bytes()
                lines = raw.decode('utf-8').splitlines()
                self.lines[name], self.kinds[name] = lines, kind
                self.hashes[str(path)] = hashlib.sha256(raw).hexdigest()
                aliases = {path.name, path.stem, re.sub(r'^\d+_', '', path.stem)}
                if number := re.match(r'^\d+(?=_)', path.name):
                    aliases.add(number[0])
                if heading := next((s[2:] for s in lines if s.startswith('# ')), None):
                    aliases.add(heading)
                for alias in aliases:
                    self.aliases[alias].append(name)

    def search_texts(self, query, kind=''):
        if kind and kind not in KINDS:
            return {'error': 'kind 不合法。', 'allowed_kinds': list(KINDS)}
        if not query:
            return {'error': '请给出非空字面搜索词。'}
        matches, counts = [], Counter()
        for name, lines in self.lines.items():
            if kind and self.kinds[name] != kind:
                continue
            section = ''
            for number, line in enumerate(lines, 1):
                if line.startswith('## '):
                    section = line[3:]
                if query in line:
                    counts[name] += 1
                    if len(matches) < 80:
                        matches.append({'file': name, 'kind': self.kinds[name],
                            'line': number, 'text': line, 'section': section,
                            'speaker': speaker_for_line(line)})
        total = sum(counts.values())
        result = {'query': query, 'kind': kind, 'matches': matches,
                  'total_matches': total, 'truncated': total > 80}
        if total > 80:
            result['top_files'] = [{'file': name, 'matches': count}
                                   for name, count in counts.most_common(20)]
            result['notice'] = '最多返回80条；结果已截断，请换更具体的词。'
        return result

    def read_text(self, file, start_line=1, end_line=120):
        choices = [file] if file in self.lines else self.aliases.get(file, [])
        if len(choices) != 1:
            return {'error': '未找到唯一文本；请使用候选的完整文件名。',
                    'candidates': choices[:10], 'total_candidates': len(choices)}
        name = choices[0]
        lines = self.lines[name]
        if start_line < 1 or end_line < start_line or start_line > len(lines):
            return {'error': '行号范围无效。', 'total_lines': len(lines)}
        end = min(end_line, start_line + 119, len(lines))
        truncated = end < min(end_line, len(lines))
        notice = '单次最多120行，请继续分段读取。' if truncated else ''
        notice += unread_notice(len(lines), end, '文本')
        return {'file': name, 'kind': self.kinds[name], 'total_lines': len(lines),
                'start_line': start_line, 'end_line': end, 'truncated': truncated,
                'notice': notice,
                'text': '\n'.join(f'L{i}: {lines[i-1]}' for i in range(start_line, end+1))}


class AgentTools(FrozenTools):
    def __init__(self, root, cache=None):
        super().__init__(root, cache)
        self.texts = Texts(root)
        self.hashes.update(self.texts.hashes)

    def tools(self):
        @tool
        def search_texts(query: str, kind: str = '') -> dict:
            """字面搜索书籍与任务以外的游戏原文，最多80条，返回文件、kind、行号、原文、小节与说话人；截断时附命中最多的文件。kind 留空搜全部，或限定 character_story、voiceline、hangout、talk_group、talk、material_type、weapon、artifact_set、anecdote、subtitle、readable、achievement、creature、costume、wings、blossom。人物台词仅表示该人物这样说。"""
            return self.texts.search_texts(query, kind)

        @tool
        def read_text(file: str, start_line: int = 1, end_line: int = 120) -> dict:
            """按搜索或候选返回的文件标识（agd_类型/文件名）、完整文件名、去数字前缀的名称、唯一编号或首个标题读取游戏原文；歧义列候选。物理行号从1起含空行，默认1–120行，一次最多120行，未读完会提示。人物台词仅表示该人物这样说。"""
            return self.texts.read_text(file, start_line, end_line)
        return super().tools() + [search_texts, read_text]
