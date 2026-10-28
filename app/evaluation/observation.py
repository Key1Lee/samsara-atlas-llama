"""Opt-in metadata-only workflow evidence. Never stores raw sensitive content."""
from contextvars import ContextVar
import os
import time
import hashlib

def digest(text):return hashlib.sha256(text.encode()).hexdigest()

active=ContextVar('atlas_observation',default=None)

def begin(stage):
    if os.getenv('ATLAS_EVAL_DIAGNOSTICS','0')!='1':return None,None
    observation={'stage':stage,'started_monotonic':time.monotonic(),'retrieval':[],'model_calls':[]}
    return observation,active.set(observation)

def end(observation,token):
    if token is None:return
    observation['elapsed_seconds']=time.monotonic()-observation.pop('started_monotonic')
    active.reset(token)

def record_retrieval(items):
    if active.get() is not None:active.get()['retrieval']=items

def record_model(prompt,model,status,**metrics):
    current=active.get()
    if current is None:return
    # No arbitrary provider fields, response text, prompts or completion reasoning.
    allowed={'prompt_eval_count','eval_count','total_duration','load_duration','eval_duration','response_sha256','response_bytes','model_version','schema_sha256','elapsed_seconds'}
    current['model_calls'].append({'prompt_sha256':digest(prompt),'prompt_version':'atlas.generation.v1',
        'model':model if isinstance(model,str) and model in {'llama3.2','llama3.2:latest'} else 'configured_local_model',
        'completion_status':status,'raw_content_retained':False,
        **{k:v for k,v in metrics.items() if k in allowed}})


def record_parse(value=None,success=False):
    if active.get() is not None:
        import json
        active.get()['parsing']={'succeeded':success,'parsed_response_sha256':digest(json.dumps(value,sort_keys=True)) if success else None,
            'parsed_response_content_retained':False}
