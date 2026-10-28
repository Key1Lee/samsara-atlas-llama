"""Run with python -m app.evaluation.cli. Operator-local, no HTTP exposure."""
import argparse
from concurrent.futures import wait, FIRST_COMPLETED
import json
from pathlib import Path
import platform
import subprocess
import time
from .artifacts import ArtifactStore
from .contracts import Evaluation
from .evidence import snapshots, CaseEvidence, digest
from .harness import OfflineQueue
from .local import Config, LocalEvaluator, EvaluationFailure
from .reporting import report, agreement

ROOT=Path(__file__).resolve().parents[2]

def dataset(path,split):
    data=json.loads(path.read_text())
    if data['version']!='atlas.calibration.v1':raise ValueError('Unsupported dataset version')
    cases=[];labels={}
    for row in data['cases']:
        if split!='all' and row['split']!=split:continue
        case=CaseEvidence(**row['case']);cases.append(case)
        labels[str(case.case_id)]=row['synthetic_expected_concerns']
    return cases,labels,data

def hardware():
    result={'platform':platform.system(),'machine':platform.machine()}
    if platform.system()=='Darwin':
        for key in ('hw.memsize','machdep.cpu.brand_string'):
            try:result[key]=subprocess.check_output(['sysctl','-n',key],text=True,timeout=3).strip()
            except (OSError,subprocess.SubprocessError):result[key]=None
    return result

def run_cases(cases,config,semantic):
    adapter=LocalEvaluator(config);queue=OfflineQueue(config,adapter);pending=set();results=[]
    try:
        for case in cases:
            while len(pending)>=config.queue_capacity:
                done,pending=wait(pending,return_when=FIRST_COMPLETED)
                results.extend(f.result() for f in done)
            f=queue.submit(case,semantic)
            if f is None:raise RuntimeError('Queue backpressure invariant violated')
            pending.add(f)
        results.extend(f.result() for f in pending)
    finally:queue.close()
    return sorted(results,key=lambda r:r.case_id)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True,help='Owner-only private artifact directory')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('probe');sub.add_parser('schema')
    d=sub.add_parser('dataset');d.add_argument('--file',type=Path,default=ROOT/'tests/fixtures/evaluation_v1.json')
    d.add_argument('--human-labels',type=Path);d.add_argument('--split',choices=['dev','test','all'],default='dev');d.add_argument('--semantic',action='store_true')
    c=sub.add_parser('cases');c.add_argument('--database',type=Path,required=True);c.add_argument('--docs',type=Path,required=True)
    c.add_argument('--case-id',type=int,action='append');c.add_argument('--employee-id');c.add_argument('--semantic',action='store_true')
    c.add_argument('--limit',type=int,default=1000)
    a=p.parse_args();store=ArtifactStore(a.output);config=Config.from_env()
    if a.command=='schema':store.write('evaluation-schema.json',Evaluation.model_json_schema());return
    if a.command=='probe':
        evaluator=LocalEvaluator(config)
        try:state={'status':'available','runtime':evaluator.preflight()}
        except EvaluationFailure as exc:state={'status':exc.status,'metrics':exc.metrics}
        store.write('probe.json',{**state,'hardware':hardware()});print(json.dumps(state));return
    labels={};dataset_info=None
    if a.command=='dataset':cases,labels,dataset_info=dataset(a.file,a.split)
    else:
        if a.limit<1:raise ValueError('Positive case limit required')
        import itertools
        cases=list(itertools.islice(snapshots(a.database,a.docs,set(a.case_id) if a.case_id else None,a.employee_id),a.limit))
    code_sha256=digest({str(p.relative_to(ROOT)):digest(p.read_text()) for p in sorted((ROOT/'app').rglob('*.py'))})
    start=time.monotonic();results=run_cases(cases,config,a.semantic)
    summary=report(results)
    summary.update({'code_sha256':code_sha256,'config':{'model':config.model,'digest':config.digest,'timeout':config.timeout,'workers':1,'queue_capacity':config.queue_capacity,'prompt_byte_limit':config.max_prompt_bytes,'output_token_limit':config.max_output_tokens},'wall_seconds':time.monotonic()-start,'hardware':hardware(),
        'mode':'deterministic_plus_gpt_oss' if a.semantic else 'deterministic_only',
        'input_snapshot_hashes':[r.metrics['snapshot_sha256'] for r in results],
        'dataset_version':dataset_info['version'] if dataset_info else None,
        'dataset_sha256':digest(dataset_info) if dataset_info else None})
    if labels:
        summary['synthetic_expectation_agreement']=agreement(results,labels)
        summary['human_label_agreement']=None
        if a.human_labels:
            provided=json.loads(a.human_labels.read_text())
            if provided.get('version')!='atlas.human-calibration.v1' or provided.get('dataset_sha256')!=digest(dataset_info):
                raise ValueError('Human labels must match the exact calibration dataset')
            from .contracts import SemanticScores
            human={}
            for row in provided['results']:
                if not row.get('reviewer_reference') or not row.get('reviewed_at') or not set(row['concerns'])<=set(SemanticScores.model_fields):
                    raise ValueError('Human label provenance or categories invalid')
                key=str(row['case_id'])
                if key in human:raise ValueError('Duplicate human label')
                human[key]=row['concerns']
            summary['human_label_agreement']=agreement(results,human)
            summary['human_labels_sha256']=digest(provided)
        summary['label_limitation']='Synthetic author expectations, not independently human-reviewed labels. No accuracy claim.'
        expected={row['case']['case_id']:row['expected_failed_checks'] for row in dataset_info['cases']}
        summary['deterministic_fixture_contracts']={
            'passed':sum(set(expected[r.case_id])=={c.code for c in r.deterministic_checks if c.status=='fail'} for r in results),
            'total':len(results)}
    for result in results:store.write(f'case-{result.case_id}.json',result.model_dump())
    store.write('summary.json',summary)
    print(json.dumps({k:summary[k] for k in ['total_cases','workflow_failures','evaluator_statuses','wall_seconds']}))

if __name__=='__main__':main()
