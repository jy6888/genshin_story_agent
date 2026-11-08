"""Audited authorization-specific settlement; preserve the old failure evidence."""
import json
from datetime import datetime,timezone
from story_agent.budget import BudgetStopped
from story_agent.strong_agent_sa1 import save

AUTHORITY='D-20261107-017; codex/110 §8.4'


def settle_estimate(ledger,identifier,run_id,value):
    """Pending or uncertain -> settled, with immutable before/estimate history."""
    with ledger.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        row=db.execute('SELECT run_id,status,actual,prompt_tokens,completion_tokens,http_status,reserved FROM calls WHERE id=?',(identifier,)).fetchone()
        db.execute('CREATE TABLE IF NOT EXISTS tl5b_empty_reconciliations (call_id TEXT PRIMARY KEY,authority TEXT,before_json TEXT,estimate_json TEXT,recorded_at TEXT)')
        previous=db.execute('SELECT estimate_json FROM tl5b_empty_reconciliations WHERE call_id=?',(identifier,)).fetchone()
        if previous:
            if json.loads(previous[0])!=value or not row or row[0]!=run_id or row[1]!='settled' or row[2]!=value['total_tokens']:
                raise BudgetStopped('Settlement history mismatch')
            return {'already_settled':True,'estimate':value}
        if not row or row[0]!=run_id or row[1] not in ('pending','uncertain') or row[2] is not None:
            raise BudgetStopped('Unexpected empty-response settlement target')
        db.execute('INSERT INTO tl5b_empty_reconciliations VALUES (?,?,?,?,?)',
            (identifier,AUTHORITY,json.dumps(list(row)),json.dumps(value,ensure_ascii=False),datetime.now(timezone.utc).isoformat()))
        db.execute("UPDATE calls SET status='settled',actual=?,prompt_tokens=?,completion_tokens=?,http_status=? WHERE id=?",
            (value['total_tokens'],value['prompt_tokens'],value['completion_tokens'],value.get('http_status',200),identifier))
    return {'already_settled':False,'before':list(row),'estimate':value}


def settle_old105(ledger,old_directory,evidence):
    from story_agent.strong_agent_sa1 import read_json,sha
    identifier=read_json(old_directory/'reservation.json')['id']
    error=read_json(old_directory/'error.json')
    if error!=dict(http_status=200,error_type='InvalidResponse',received_bytes=4,error_body='null'):
        raise BudgetStopped('Old105 failure differs from authorization')
    value=dict(topic='T5',call_id=identifier,prompt_tokens=848,completion_tokens=49152,total_tokens=50000,
        prompt_cache_miss_tokens=848,budget_tokens=50000,estimated=True,http_status=200,
        authority=AUTHORITY,method='Explicit authorized conservative 50000 settlement; billing estimate, not provider-measured raw usage')
    result=settle_estimate(ledger,identifier,'TL5b:v0.94.1:stage1:T5',value)
    record=dict(**result,authority=AUTHORITY,source=str(old_directory),
        source_hashes={p.name:sha(p) for p in old_directory.iterdir() if p.is_file()},provider_calls=0)
    if evidence.exists():
        saved=read_json(evidence)
        if saved['estimate']!=value:raise BudgetStopped('Existing105 settlement evidence differs')
        return saved
    save(evidence,record)
    return record
