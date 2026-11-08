"""SA1 v0.81.0: three read-only book tools and a metered Deep Agent.

No answer schema, semantic validator, feedback, or research stage controller.
The transport guards accounting; the corpus capability has no arbitrary IO.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import threading

import httpx
from langchain_core.tools import tool
from story_agent.budget import BudgetStopped, TokenLedger

VERSION = 'v0.81.0'
MODEL = 'gpt-6-astra'
CAP = 500_000
SYSTEM_PROMPT = (
    '你在做原神书籍考据，只依据工具提供的书籍原文作答，不使用你自己已知的原神设定。'
    '结论写明依据的书和大致位置（书名与行号即可，不需要逐字引文）。'
    '可以说“没找到”或“不确定”，信息不全是正常的。用中文自由撰写报告。'
    '报告最后单列一段：还想查但查不到，或想做但没有工具做的事。'
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def emit(**value):
    print(json.dumps(value, ensure_ascii=False), flush=True)


class Books:
    """Load only the pinned agd_book directory and manifest-allowlisted maps."""
    def __init__(self, root):
        root = Path(root).resolve()
        corpus = (root / '.runtime/istaroth-checkpoint/text/agd_book').resolve()
        maps = (root / '.runtime/book-map/v1').resolve()
        manifest = root / 'data/manifests/book-map-v1.json'
        self.hashes = {str(manifest): sha(manifest)}
        self.lines, self.aliases, self.maps = {}, {}, {}
        for path in sorted(corpus.glob('*.txt')):
            if path.is_symlink() or path.resolve().parent != corpus:
                raise ValueError('Book outside fixed corpus')
            title = re.sub(r'^\d+_', '', path.stem)
            # Full filename is unambiguous even when corpus titles collide.
            self.lines[path.name] = path.read_text(encoding='utf-8').splitlines()
            self.aliases.setdefault(title, []).append(path.name)
            self.hashes[str(path)] = sha(path)
        for entry in read_json(manifest)['books']:
            source, mapped = Path(entry['source_path']), Path(entry['path'])
            source = source if source.is_absolute() else root / source
            mapped = mapped if mapped.is_absolute() else root / mapped
            if (source.resolve().parent != corpus or mapped.resolve().parent != maps
                    or source.is_symlink() or mapped.is_symlink()):
                raise ValueError('Manifest outside allowed book/map roots')
            if sha(source) != entry['source_sha256'] or sha(mapped) != entry['sha256']:
                raise ValueError('Pinned map/source hash mismatch')
            self.maps[source.name] = read_json(mapped)
            self.hashes[str(mapped)] = sha(mapped)

    def resolve(self, book):
        if book in self.lines:
            return book
        choices = self.aliases.get(book, [])
        return choices[0] if len(choices) == 1 else None

    def book_map(self, book=''):
        if book:
            name = self.resolve(book)
            if name not in self.maps:
                return {'error': '这本书没有书图条目；可用 search_books 查书文或核对书名。'}
            return self.maps[name]
        rows = []
        for name, item in self.maps.items():
            gists = item['targets'].get('T4', {}).get('gists', [])
            summary = ' '.join(g['text'] for g in gists[:2])
            if not summary:
                # Existing map text only; never invent a new synopsis.
                summary = '书图无总览；条目摘录：' + '；'.join(
                    x['text'] for x in item['targets'].get('T1', {}).get('items', [])[:2])
            rows.append({'book': re.sub(r'^\d+_', '', Path(name).stem),
                         'filename': name, 'summary': summary})
        return {'books': rows, 'map_count': len(rows), 'searchable_books': len(self.lines)}

    def search_books(self, query):
        if not query:
            return {'error': '请给出非空字面搜索词。'}
        matches = []
        total = 0
        for name, lines in self.lines.items():
            for number, line in enumerate(lines, 1):
                if query in line:
                    total += 1
                    if len(matches) < 80:
                        matches.append({'book': name, 'line': number, 'text': line})
        return {'query': query, 'matches': matches, 'total_matches': total,
                'truncated': total > 80,
                'notice': '最多返回80条；结果已截断，请细化搜索词。' if total > 80 else ''}

    def read_book(self, book, start_line=1, end_line=120):
        name = self.resolve(book)
        if name is None:
            return {'error': '未找到唯一书名；请用 search_books 返回的完整文件名。'}
        lines = self.lines[name]
        if start_line < 1 or end_line < start_line or start_line > len(lines):
            return {'error': '行号范围无效。', 'total_lines': len(lines)}
        end = min(end_line, start_line + 119, len(lines))
        return {'book': name, 'total_lines': len(lines), 'start_line': start_line,
                'end_line': end, 'truncated': end < min(end_line, len(lines)),
                'notice': '单次最多120行，请继续分段读取。' if end < min(end_line, len(lines)) else '',
                'text': '\n'.join(f'L{i}: {lines[i-1]}' for i in range(start_line, end+1))}

    def tools(self):
        @tool
        def book_map(book: str = '') -> dict:
            """无书名时列出63本书的摘要；给精确书名或文件名时读该书完整书图。"""
            return self.book_map(book)

        @tool
        def search_books(query: str) -> dict:
            """在全部游戏书籍原文中做字面子串搜索，返回书名、物理行号和文本，最多80条。"""
            return self.search_books(query)

        @tool
        def read_book(book: str, start_line: int = 1, end_line: int = 120) -> dict:
            """按精确书名或文件名读取原文（从1开始的物理行号，含空行），一次最多120行。"""
            return self.read_book(book, start_line, end_line)

        return [book_map, search_books, read_book]


def build_agent(model, books):
    from deepagents import create_deep_agent
    from deepagents.profiles import (
        GeneralPurposeSubagentProfile, HarnessProfile, register_harness_profile,
    )
    # Use the installed framework's public configuration, not custom middleware.
    # Optional built-in notebook tools are omitted: SA1 exposes exactly three.
    register_harness_profile('openai:' + MODEL, HarnessProfile(
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        excluded_tools={'ls', 'read_file', 'write_file', 'edit_file', 'delete', 'glob', 'grep', 'execute'},
    ))
    return create_deep_agent(model=model, tools=books.tools(), system_prompt=SYSTEM_PROMPT)


def usage_for_ledger(raw):
    if not isinstance(raw, dict):
        return None
    return {'prompt_tokens': raw.get('input_tokens'), 'completion_tokens': raw.get('output_tokens'),
            'total_tokens': raw.get('total_tokens')}


class MeteredResponses(httpx.BaseTransport):
    """Exactly-once, non-streaming Responses transport; no application retries."""
    def __init__(self, out, topic, key, inner=None):
        self.out, self.topic, self.key = Path(out), topic, key
        self.ledger = TokenLedger(self.out / 'ledger/token_budget.sqlite')
        self.inner = inner or httpx.HTTPTransport(retries=0)
        self.lock = threading.Lock()

    def safe(self, value):
        if self.key in json.dumps(value, ensure_ascii=False):
            raise BudgetStopped('Credential echo rejected before persistence')
        return value

    def handle_request(self, request):
        with self.lock:
            return self._request(request)

    def _request(self, request):
        if (request.method != 'POST' or str(request.url) != 'https://api.openai.com/v1/responses'):
            raise BudgetStopped('Only official Responses POST allowed')
        payload = self.safe(json.loads(request.content))
        if (payload.get('model') != MODEL or payload.get('reasoning', {}).get('effort') != 'high'
                or payload.get('stream') or payload.get('store') is not False):
            raise BudgetStopped('Unexpected model/transport settings')
        names = {t.get('name') for t in payload.get('tools', [])}
        if names - {'book_map', 'read_book', 'search_books'}:
            raise BudgetStopped('Unexpected tool capability: ' + ', '.join(sorted(str(n) for n in names)))
        summary = self.ledger.summary()
        if set(summary['status_counts']) - {'settled'}:
            raise BudgetStopped('Unsettled ledger; no blind resend')
        # Same conservative UTF-8 reservation as OA2, with a hard remaining cap.
        output_bound = payload.get('max_output_tokens')
        if not isinstance(output_bound, int) or output_bound <= 0:
            raise BudgetStopped('Missing output accounting bound')
        upper = 2 * len(request.content) + 2048 + output_bound
        if summary['accounted_tokens'] + upper > CAP:
            raise BudgetStopped('Remaining token cap cannot cover next request bound')
        number = summary['calls'] + 1
        directory = self.out / self.topic / 'calls' / f'{number:03d}'
        save(directory / 'request.json', payload)
        ident = self.ledger.reserve('SA1:' + VERSION + ':' + self.topic, MODEL, upper)
        save(directory / 'reservation.json', {'call_id': ident, 'upper': upper})
        emit(request_started=number, topic=self.topic, actual_tokens=summary['accounted_tokens'])
        try:
            response = self.inner.handle_request(request)
            response.read()
        except Exception:
            self.ledger.settle(ident, None, None)
            raise BudgetStopped('Transport failure') from None
        try:
            body = self.safe(response.json())
        except Exception:
            self.ledger.settle(ident, None, response.status_code)
            raise BudgetStopped('Unsafe or invalid JSON response') from None
        save(directory / 'response.json', body)
        usage = body.get('usage') if response.is_success else None
        self.ledger.settle(ident, usage_for_ledger(usage), response.status_code)
        save(directory / 'usage.json', {'usage': usage, 'ledger': self.ledger.summary()})
        emit(request_settled=number, topic=self.topic, usage=usage,
             actual_tokens=self.ledger.summary()['accounted_tokens'])
        if body.get('status') != 'completed':
            raise BudgetStopped('Non-completed API response; no retry')
        return response

    def close(self):
        self.inner.close()


def live_model(key, transport):
    from langchain_openai import ChatOpenAI
    from story_agent.budget import DenyAsyncTransport
    return ChatOpenAI(model=MODEL, api_key=key, base_url='https://api.openai.com/v1',
        use_responses_api=True, reasoning={'effort': 'high'}, store=False,
        include=['reasoning.encrypted_content'], output_version='responses/v1',
        max_tokens=49152, max_retries=0, streaming=False,
        http_client=httpx.Client(transport=transport, trust_env=False, follow_redirects=False,
                                timeout=httpx.Timeout(900, connect=20)),
        http_async_client=httpx.AsyncClient(transport=DenyAsyncTransport(), trust_env=False))


def summarize_usage(out):
    rows = [read_json(p)['usage'] for p in Path(out).glob('*/calls/*/usage.json')]
    return {'requests': len(rows), 'input_tokens': sum(u['input_tokens'] for u in rows),
            'cached_tokens': sum(u.get('input_tokens_details', {}).get('cached_tokens', 0) for u in rows),
            'output_tokens': sum(u['output_tokens'] for u in rows),
            'reasoning_tokens': sum(u.get('output_tokens_details', {}).get('reasoning_tokens', 0) for u in rows),
            'total_tokens': sum(u['total_tokens'] for u in rows)}
