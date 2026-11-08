"""D-20261107-007 estimate settlement only. Never resend A1/001."""
import json
from pathlib import Path
import sys

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE))
from story_agent.pt0_wire_v921 import LEGACY_ID,LEGACY_RUN,LEGACY_BUDGET
from story_agent.strong_agent_sa1 import read_json,save,sha,emit
from story_agent.strong_agent_sa2 import DB
from story_agent.budget import TokenLedger

TASK=CODE/'docs/project/tasks/T-20261107-002'
OLD=TASK/'evidence/pt0-v0.92.0/stage1'
OUT=TASK/'evidence/pt0-v0.92.1/stage1'
SPEC=CODE/'docs/design/codex/108-pt0-external-request.md'
AUTHORITY='D-20261107-007; codex/108 §11; user explicit recovery and cf40c34 correction'


def value():
    p=read_json(OLD/'A1/calls/001/operator-interruption-review.json')
    assert p['reservation'][0]==LEGACY_ID and p['reservation'][4]==LEGACY_BUDGET
    assert sha(OLD/'A1/calls/001/request-body.bin')==p['request_sha256']
    assert not (OLD/'A1/calls/001/response.json').exists()
    return dict(p['proposed_estimate'],authority=AUTHORITY,topic='INTERRUPTED_A1',call_id=LEGACY_ID,
                operator_interruption=True,provider_usage_unknown=True,no_resend=True)


def settle(ledger,directory,estimate):
    """Transaction retains the before row and labels actual-column value estimated."""
    directory=Path(directory)
    with ledger.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        row=db.execute('SELECT run_id,status,actual,reserved,prompt_tokens,completion_tokens,http_status FROM calls WHERE id=?',(LEGACY_ID,)).fetchone()
        db.execute('CREATE TABLE IF NOT EXISTS pt0_operator_estimates (call_id TEXT PRIMARY KEY,authority TEXT NOT NULL,before_json TEXT NOT NULL,estimate_json TEXT NOT NULL)')
        audit=db.execute('SELECT authority,before_json,estimate_json FROM pt0_operator_estimates WHERE call_id=?',(LEGACY_ID,)).fetchone()
        if audit:
            assert audit[0]==AUTHORITY and json.loads(audit[2])==estimate
            assert row==(LEGACY_RUN,'settled',60262,60262,11110,49152,None)
            before=json.loads(audit[1])
        else:
            assert row==(LEGACY_RUN,'pending',None,60262,None,None,None),row
            before=list(row)
            db.execute('INSERT INTO pt0_operator_estimates VALUES (?,?,?,?)',
                (LEGACY_ID,AUTHORITY,json.dumps(before),json.dumps(estimate,ensure_ascii=False)))
            db.execute("UPDATE calls SET status='settled',actual=60262,prompt_tokens=11110,completion_tokens=49152 WHERE id=? AND status='pending'",(LEGACY_ID,))
        after=db.execute('SELECT run_id,status,actual,reserved,prompt_tokens,completion_tokens,http_status FROM calls WHERE id=?',(LEGACY_ID,)).fetchone()
    evidence={'authority':AUTHORITY,'before':before,'after':list(after),'estimate':estimate,
        'provider_actual_usage_known':False,'estimate_counted':60262,'resend_requests':0,
        'original_A1_directory_unchanged':True,'original_user_stop_retained':True}
    for path,record in ((directory/'estimates'/f'{LEGACY_ID}.json',estimate),
                        (directory/'legacy-estimate-settlement.json',evidence)):
        if path.exists():assert read_json(path)==record
        else:save(path,record)
    return evidence


def main():
    spec=SPEC.read_text(encoding='utf-8')
    assert 'D-20261107-007' in spec and '60,262' in spec and '不再重发' in spec
    assert '依 D-20261107-007 解除' in (TASK/'PROGRESS.md').read_text(encoding='utf-8')
    assert (OLD/'user-stop.json').exists()
    estimate=value()
    result=settle(TokenLedger(DB),OUT,estimate)
    emit(estimated_settlement_applied=True,estimate=result['estimate_counted'],resend_requests=0)


if __name__=='__main__':main()
