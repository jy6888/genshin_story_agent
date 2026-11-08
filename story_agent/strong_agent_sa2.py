"""SA2 v0.82.0: SA1 capabilities with DeepSeek thinking and shared accounting."""
from collections import Counter
import json
import os
from pathlib import Path
import threading
from urllib.parse import urlsplit

import httpx
from langchain_deepseek import ChatDeepSeek

from story_agent.budget import BudgetStopped, BudgetTransport, DenyAsyncTransport, TokenLedger
from story_agent.strong_agent_sa1 import Books, SYSTEM_PROMPT, emit, read_json, save, sha

VERSION = 'v0.82.0'
MODEL = 'deepseek-v4-pro'
CAP = 3_000_000
PREFIX = 'SA2:' + VERSION + ':'
ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / 'data/runtime/token_budget.sqlite'


def usage(ledger):
    with ledger.connect() as db:
        rows = db.execute('SELECT status, actual, reserved FROM calls WHERE run_id LIKE ?',
                          (PREFIX + '%',)).fetchall()
    return {'requests': len(rows), 'total_tokens': sum(r[1] if r[1] is not None else r[2] for r in rows),
            'status_counts': dict(Counter(r[0] for r in rows))}


class ThinkingDeepSeek(ChatDeepSeek):
    """Installed LangChain saves reasoning but its OpenAI serializer omits it."""
    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        messages = self._convert_input(input_).to_messages()
        payload = super()._get_request_payload(messages, stop=stop, **kwargs)
        if len(messages) != len(payload['messages']):
            raise BudgetStopped('Unexpected message serialization')
        for original, wire in zip(messages, payload['messages']):
            if original.type == 'ai':
                reasoning = original.additional_kwargs.get('reasoning_content')
                if not isinstance(reasoning, str):
                    raise BudgetStopped('Missing reasoning for assistant history')
                wire['reasoning_content'] = reasoning
        return payload


def build_agent(model, books):
    from deepagents import create_deep_agent
    from deepagents.profiles import GeneralPurposeSubagentProfile, HarnessProfile, register_harness_profile
    # Identical SA1 profile, registered for the new provider/model key.
    register_harness_profile('deepseek:' + MODEL, HarnessProfile(
        general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        excluded_tools={'ls', 'read_file', 'write_file', 'edit_file', 'delete', 'glob', 'grep', 'execute'},
    ))
    return create_deep_agent(model=model, tools=books.tools(), system_prompt=SYSTEM_PROMPT)


def credentials():
    from dotenv import dotenv_values
    names = ('OPENAI_API_KEY', 'OPENAI_BASE_URL')
    config = {name: os.environ[name] for name in names if name in os.environ}
    if len(config) < len(names):
        local = dotenv_values(ROOT / '.env', encoding='utf-8', interpolate=False)
        config.update({name: local[name] for name in names if name not in config and name in local})
    endpoint = config.get('OPENAI_BASE_URL', '').rstrip('/')
    url = urlsplit(endpoint)
    key = config.get('OPENAI_API_KEY')
    if (url.scheme != 'https' or url.hostname != 'api.deepseek.com' or url.username
            or url.password or url.query or url.fragment or not key):
        raise BudgetStopped('Authorized DeepSeek configuration unavailable')
    return endpoint, key


class MeteredDeepSeek(BudgetTransport):
    def __init__(self, out, topic, key, ledger=None, inner=None):
        self.out, self.topic, self.key = Path(out), topic, key
        self.lock = threading.Lock()
        super().__init__(ledger or TokenLedger(DB), PREFIX + topic, inner=inner)

    def safe(self, value):
        if self.key in json.dumps(value, ensure_ascii=False):
            raise BudgetStopped('Credential echo rejected before persistence')
        return value

    def handle_request(self, request):
        with self.lock:
            return self._request(request)

    def _request(self, request):
        payload = self.safe(json.loads(request.content))
        if (request.url.scheme != 'https' or request.url.host != 'api.deepseek.com'
                or payload.get('model') != MODEL or payload.get('thinking') != {'type': 'enabled'}
                or payload.get('reasoning_effort') != 'max' or 'temperature' in payload
                or payload.get('max_tokens', payload.get('max_completion_tokens')) != 49152
                or payload.get('stream')):
            raise BudgetStopped('Unexpected model/transport settings')
        if {t['function']['name'] for t in payload.get('tools', [])} != {'book_map', 'search_books', 'read_book'}:
            raise BudgetStopped('Unexpected tools')
        summary = self.ledger.summary()
        if summary['limit'] is not None or set(summary['status_counts']) - {'settled'}:
            raise BudgetStopped('Shared ledger must be accounting-only and fully settled')
        used = usage(self.ledger)
        upper = 2 * len(request.content) + 2048 + 49152
        if used['total_tokens'] + upper > CAP:
            raise BudgetStopped('Remaining task cap cannot cover next request bound')
        number = used['requests'] + 1
        directory = self.out / self.topic / 'calls' / f'{number:03d}'
        save(directory / 'request.json', payload)
        emit(request_started=number, topic=self.topic, actual_tokens=used['total_tokens'])
        response = super().handle_request(request)
        body = self.safe(response.json())
        save(directory / 'response.json', body)
        save(directory / 'usage.json', {'usage': body.get('usage'), 'task_usage': usage(self.ledger)})
        choice = body['choices'][0]
        message = choice['message']
        save(directory / 'message.json', message)
        reasoning = message.get('reasoning_content')
        if not isinstance(reasoning, str) or not reasoning.strip():
            raise BudgetStopped('Missing reasoning output')
        (directory / 'reasoning.txt').write_text(reasoning, encoding='utf-8')
        (directory / 'content.txt').write_text(message.get('content') or '', encoding='utf-8')
        emit(request_settled=number, topic=self.topic, usage=body.get('usage'),
             reasoning_characters=len(reasoning), actual_tokens=usage(self.ledger)['total_tokens'])
        if choice.get('finish_reason') not in ('stop', 'tool_calls'):
            raise BudgetStopped('Non-completed API response; no retry')
        return response


def live_model(key, transport, endpoint='https://api.deepseek.com/v1'):
    return ThinkingDeepSeek(model=MODEL, api_key=key, api_base=endpoint,
        temperature=None, reasoning_effort='max', extra_body={'thinking': {'type': 'enabled'}},
        max_tokens=49152, max_retries=0, streaming=False,
        http_client=httpx.Client(transport=transport, trust_env=False, follow_redirects=False,
                                timeout=httpx.Timeout(900, connect=20)),
        http_async_client=httpx.AsyncClient(transport=DenyAsyncTransport(), trust_env=False))


def summarize_usage(out):
    rows = [read_json(p)['usage'] for p in Path(out).glob('*/calls/*/usage.json')]
    reasoning = [u.get('completion_tokens_details', {}).get('reasoning_tokens') for u in rows]
    return {'requests': len(rows), 'input_tokens': sum(u['prompt_tokens'] for u in rows),
            'cached_tokens': sum(u.get('prompt_cache_hit_tokens', 0) for u in rows),
            'output_tokens': sum(u['completion_tokens'] for u in rows),
            'reasoning_tokens': sum(reasoning) if all(isinstance(n, int) for n in reasoning) else None,
            'reasoning_tokens_note': 'null means provider did not expose a separate count; included in completion/total',
            'total_tokens': sum(u['total_tokens'] for u in rows)}
