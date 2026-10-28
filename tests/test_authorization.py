"""BR-01: existing bearer-to-employee policy, enforced before records or side effects."""
import hashlib
import json
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.storage.models import Case, CaseNote, Step, EmployeeRecord


@pytest.fixture
def subjects(db_session):
    result = []
    for _ in range(2):
        employee = 'AUTH-' + uuid4().hex
        token = uuid4().hex
        db_session.add(EmployeeRecord(employee_id=employee, metadata_json='{}', last_activities_json='[]'))
        case = Case(status='completed', inputs_json=json.dumps({'employee_id':employee,
            'case_transcript':'PRIVATE-' + employee, 'tone':'direct'}), outputs_json='{}')
        db_session.add(case)
        db_session.flush()
        db_session.add(CaseNote(case_id=case.id, employee_id=employee, note='PRIVATE-NOTE-' + employee))
        result.append({'employee':employee,'token':token,'case':case.id})
    db_session.commit()
    return result


def configure(monkeypatch, subjects):
    monkeypatch.setenv('ATLAS_ENV','production')
    monkeypatch.setenv('ATLAS_EMPLOYEE_TOKEN_HASHES',json.dumps({
        s['employee']:hashlib.sha256(s['token'].encode()).hexdigest() for s in subjects}))


def request_body(employee):
    return {'inputs':{'employee_id':employee,'case_transcript':'How does PTO accrue?','tone':'direct'},
            'idempotency_key':uuid4().hex}


def snapshot(db):
    return {
        model.__tablename__: [tuple(getattr(row,c.name) for c in model.__table__.columns)
                              for row in db.query(model).order_by(model.id)]
        for model in (Case,Step,CaseNote)}


@pytest.mark.parametrize('mode',[None,'','prod','misspelled'])
def test_absent_or_unknown_mode_never_implicitly_enables_anonymous_access(db_session,monkeypatch,subjects,mode):
    if mode is None: monkeypatch.delenv('ATLAS_ENV',raising=False)
    else: monkeypatch.setenv('ATLAS_ENV',mode)
    target=subjects[1]
    before=snapshot(db_session)
    client=TestClient(app)
    headers={'X-Employee-ID':target['employee']}
    responses=[client.get('/employees',headers=headers),client.get('/cases',headers=headers),
               client.get(f"/cases/{target['case']}",headers=headers),
               client.post('/cases',headers=headers,json=request_body(target['employee'])),
               client.post(f"/cases/{target['case']}/replay",headers=headers)]
    assert all(r.status_code==503 for r in responses)
    assert snapshot(db_session)==before


@pytest.mark.parametrize('authorization',[None,'Bearer incorrect','Basic incorrect'])
def test_unauthenticated_or_invalid_credentials_deny_all_operations(db_session,monkeypatch,subjects,authorization,caplog):
    configure(monkeypatch,subjects)
    target=subjects[1]
    headers={'X-Employee-ID':target['employee']}
    if authorization: headers['Authorization']=authorization
    client=TestClient(app)
    before=snapshot(db_session)
    responses=[client.get('/employees',headers=headers),client.get('/cases',headers=headers),
        client.get(f"/cases/{target['case']}",headers=headers),
        client.post('/cases',headers=headers,json=request_body(target['employee'])),
        client.post(f"/cases/{target['case']}/replay",headers=headers)]
    assert all(r.status_code==401 for r in responses)
    assert snapshot(db_session)==before
    assert all('PRIVATE-' not in r.text for r in responses)
    assert all(s['token'] not in caplog.text for s in subjects)


@pytest.mark.parametrize('attack',['header','query','case','create','replay'])
def test_authenticated_requester_cannot_select_a_different_target(db_session,monkeypatch,subjects,attack,caplog):
    configure(monkeypatch,subjects)
    requester,target=subjects
    headers={'Authorization':'Bearer '+requester['token']}
    before=snapshot(db_session)
    client=TestClient(app)
    if attack=='header': response=client.get('/cases',headers={**headers,'X-Employee-ID':target['employee']})
    elif attack=='query': response=client.get('/cases',headers=headers,params={'employee_id':target['employee']})
    elif attack=='case': response=client.get(f"/cases/{target['case']}",headers=headers)
    elif attack=='create': response=client.post('/cases',headers=headers,json=request_body(target['employee']))
    else: response=client.post(f"/cases/{target['case']}/replay",headers=headers)
    assert response.status_code in (403,404)
    assert snapshot(db_session)==before
    assert target['employee'] not in response.text
    assert 'PRIVATE-' not in response.text+caplog.text
    assert requester['token'] not in caplog.text


@pytest.mark.parametrize('bad_hash',['g'*64,'é'*64,'a'*63])
def test_invalid_credential_configuration_is_safe(db_session,monkeypatch,subjects,bad_hash):
    monkeypatch.setenv('ATLAS_ENV','production')
    monkeypatch.setenv('ATLAS_EMPLOYEE_TOKEN_HASHES',json.dumps({subjects[0]['employee']:bad_hash}))
    response=TestClient(app,raise_server_exceptions=False).get('/cases',headers={'Authorization':'Bearer anything'})
    assert response.status_code==503


def test_authorized_workflow_history_and_replay_remain_self_scoped(db_session,monkeypatch,subjects):
    configure(monkeypatch,subjects)
    a,b=subjects
    client=TestClient(app)
    headers={'Authorization':'Bearer '+a['token'],'X-Requester-ID':b['employee']}
    assert client.get('/employees',headers=headers).json()['employees']==[a['employee']]
    own=client.get('/cases',headers=headers).json()
    assert own['case_ids']==[a['case']]
    detail=client.get(f"/cases/{a['case']}",headers=headers).json()
    assert len(detail['case_notes'])==1
    assert all(note['employee_id']==a['employee'] for note in detail['case_notes'])
    assert b['employee'] not in json.dumps(detail)
    new=client.post('/cases',headers=headers,json=request_body(a['employee']))
    assert new.status_code==200 and new.json()['status']=='completed'
    new_detail=client.get(f"/cases/{new.json()['case_id']}",headers=headers).json()
    assert len(new_detail['case_notes'])==1
    assert new_detail['case_notes'][0]['employee_id']==a['employee']
    replay=client.post(f"/cases/{new.json()['case_id']}/replay",headers=headers)
    assert replay.status_code==200
    assert replay.json()['case_id']!=new.json()['case_id']
    assert b['case'] not in client.get('/cases',headers=headers).json()['case_ids']


def test_explicit_demo_does_not_override_configured_credentials(db_session,monkeypatch,subjects):
    configure(monkeypatch,subjects)
    monkeypatch.setenv('ATLAS_ENV','development')
    assert TestClient(app).get('/cases',headers={'X-Employee-ID':subjects[0]['employee']}).status_code==401
