"""A durable, fail-closed token gate at the model HTTP boundary."""
import json
from pathlib import Path
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from contextlib import contextmanager

import httpx


class BudgetStopped(RuntimeError):
    pass


class TokenLedger:
    def __init__(self, path: Path, limit: int | None = None):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS budget (id INTEGER PRIMARY KEY CHECK(id=1), token_limit INTEGER NOT NULL)')
            db.execute('INSERT OR IGNORE INTO budget VALUES (1, ?)', (limit if limit is not None else 0,))
            saved = db.execute('SELECT token_limit FROM budget WHERE id=1').fetchone()[0]
            if limit is not None and saved != limit:
                raise BudgetStopped('Stored budget differs; changing limits requires an explicit migration')
            db.execute('''CREATE TABLE IF NOT EXISTS calls (
                id TEXT PRIMARY KEY, run_id TEXT NOT NULL, model TEXT NOT NULL,
                reserved INTEGER NOT NULL, status TEXT NOT NULL, actual INTEGER,
                prompt_tokens INTEGER, completion_tokens INTEGER,
                created REAL NOT NULL, http_status INTEGER)''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def remove_limit(self, decision: str):
        """Explicit, history-preserving migration; zero denotes accounting-only mode."""
        if not decision.strip():
            raise ValueError('An authorization reference is required')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute('SELECT token_limit FROM budget WHERE id=1').fetchone()[0]
            db.execute('CREATE TABLE IF NOT EXISTS limit_history (decision TEXT PRIMARY KEY, old_limit INTEGER, new_limit INTEGER, accounted_tokens INTEGER, calls INTEGER, recorded_at TEXT)')
            if old:
                used,calls=db.execute('SELECT COALESCE(SUM(COALESCE(actual,reserved)),0),COUNT(*) FROM calls').fetchone()
                db.execute('INSERT INTO limit_history (decision,old_limit,new_limit,accounted_tokens,calls,recorded_at) VALUES (?,?,?,?,?,?)',
                           (decision,old,0,used,calls,datetime.now(timezone.utc).isoformat()))
                db.execute('UPDATE budget SET token_limit=0 WHERE id=1')

    def reserve(self, run_id: str, model: str, upper: int, max_calls: int | None = None, token_ceiling: int | None = None):
        if upper <= 0:
            raise ValueError('Reservation must be positive')
        identifier = str(uuid.uuid4())
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            uncertain = db.execute("SELECT COUNT(*) FROM calls WHERE status IN ('pending','uncertain')").fetchone()[0]
            if uncertain:
                raise BudgetStopped('Unreconciled request exists; stop before another paid call')
            count = db.execute('SELECT COUNT(*) FROM calls WHERE run_id=?', (run_id,)).fetchone()[0]
            if max_calls is not None and count >= max_calls:
                raise BudgetStopped('Per-run model call limit reached')
            used = db.execute('SELECT COALESCE(SUM(COALESCE(actual,reserved)),0) FROM calls').fetchone()[0]
            limit = db.execute('SELECT token_limit FROM budget WHERE id=1').fetchone()[0]
            unlimited = limit == 0
            if not unlimited and token_ceiling is not None:
                limit = min(limit, token_ceiling)
            if not unlimited and used + upper > limit:
                raise BudgetStopped('Token reservation would exceed the approved total')
            db.execute('INSERT INTO calls (id,run_id,model,reserved,status,created) VALUES (?,?,?,?,?,?)',
                       (identifier,run_id,model,upper,'pending',time.time()))
        return identifier

    def settle(self, identifier, usage, http_status):
        required = ('prompt_tokens','completion_tokens','total_tokens')
        valid = isinstance(usage,dict) and all(type(usage.get(k)) is int and usage[k]>=0 for k in required)
        valid = valid and usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens']
        with self.connect() as db:
            unlimited = db.execute('SELECT token_limit FROM budget WHERE id=1').fetchone()[0] == 0
            row = db.execute('SELECT reserved,status FROM calls WHERE id=?',(identifier,)).fetchone()
            if not row or row[1] != 'pending':
                raise BudgetStopped('Invalid request reconciliation')
            if not valid:
                db.execute("UPDATE calls SET status='uncertain', http_status=? WHERE id=?",(http_status,identifier))
            else:
                status = 'settled' if unlimited or usage['total_tokens']<=row[0] else 'uncertain'
                db.execute('UPDATE calls SET status=?,actual=?,prompt_tokens=?,completion_tokens=?,http_status=? WHERE id=?',
                           (status,usage['total_tokens'],usage['prompt_tokens'],usage['completion_tokens'],http_status,identifier))
        if not valid or (not unlimited and usage['total_tokens']>row[0]):
            raise BudgetStopped('Missing usage or reservation under-estimate; paid work stopped')

    def summary(self):
        with self.connect() as db:
            rows = db.execute('SELECT status,COUNT(*),SUM(COALESCE(actual,reserved)) FROM calls GROUP BY status').fetchall()
            limit = db.execute('SELECT token_limit FROM budget WHERE id=1').fetchone()[0]
            return {'limit':limit or None, 'mode':'finite' if limit else 'accounting_only',
                    'calls':sum(r[1] for r in rows), 'accounted_tokens':sum(r[2] for r in rows),
                    'status_counts':{r[0]:r[1] for r in rows}}


class BudgetTransport(httpx.BaseTransport):
    def __init__(self, ledger: TokenLedger, run_id: str, inner=None, token_ceiling=None, max_calls=None):
        self.ledger, self.run_id = ledger, run_id
        self.inner = inner or httpx.HTTPTransport(retries=0)
        self.token_ceiling = token_ceiling
        self.max_calls = max_calls

    def handle_request(self, request):
        if request.method != 'POST' or not request.url.path.endswith('/chat/completions'):
            raise BudgetStopped('Only the audited chat-completions route is enabled')
        payload = json.loads(request.content)
        output_bound = payload.get('max_tokens',payload.get('max_completion_tokens'))
        unlimited = self.ledger.summary()['limit'] is None
        if payload.get('stream') or (not unlimited and (type(output_bound) is not int or not 1 <= output_bound <= 4096)):
            raise BudgetStopped('A non-streaming output token bound is required')
        if output_bound is not None and (type(output_bound) is not int or output_bound <= 0):
            raise BudgetStopped('Invalid output token setting')
        # UTF-8 serialized request bytes conservatively cover input text/tool schemas.
        # Extra allowance covers chat framing; verify against returned usage each time.
        upper = 2 * len(request.content) + 2048 + (output_bound or 0)
        identifier = self.ledger.reserve(self.run_id,payload['model'],upper,max_calls=self.max_calls,token_ceiling=self.token_ceiling)
        try:
            response = self.inner.handle_request(request)
            response.read()
        except Exception:
            self.ledger.settle(identifier,None,None)
            raise
        try:
            usage = response.json().get('usage') if response.is_success else None
        except (ValueError, AttributeError):
            usage = None
        self.ledger.settle(identifier,usage,response.status_code)
        print(json.dumps({'model_request':identifier,'model':payload['model'],'tokens':usage['total_tokens'],
                          'budget':self.ledger.summary()},ensure_ascii=False),flush=True)
        return response

    def close(self):
        self.inner.close()


class DenyAsyncTransport(httpx.AsyncBaseTransport):
    async def handle_async_request(self, request):
        raise BudgetStopped('Async model calls are disabled until audited by the shared ledger')
