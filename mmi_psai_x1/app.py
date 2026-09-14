from __future__ import annotations
import asyncio, json, os, uuid
from datetime import datetime, timezone
from typing import Any
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

CORE = '''You are an investigator inside MMI-PSAI X1.
Method: PROBLEM -> CONNECTION -> ROOT -> AI -> TEST -> EVIDENCE -> JUDGE.
Research loop: OBSERVE -> RECORD -> VERIFY -> REPRODUCE -> CHALLENGE -> HYPOTHESIZE -> ATTACK -> INDEPENDENT TEST -> UPDATE -> RE-JUDGE.
Rules: separate observations from assumptions; generate competing hypotheses; every important hypothesis needs a falsifier; a model statement is not independent evidence; do not invent evidence; state uncertainty; prefer reproducible tests; produce a current judgement, not confidence theater.'''

ANALYZE = '''Investigate the problem below. Return ONLY JSON with keys observations, connections, hypotheses, proposed_tests, evidence_requirements, uncertainties.
Hypotheses must be objects with id, statement, supporting_observations, missing_evidence, falsifiers, prior_plausibility.
PROBLEM:\n{problem}'''

CRITIQUE = '''Attack this candidate investigation. Return ONLY JSON with keys attacks, unsupported_claims, missing_tests, best_counter_hypothesis, additional_evidence_needed.
PROBLEM:\n{problem}\nCANDIDATE:\n{candidate}\nOTHER CANDIDATES:\n{others}'''

JUDGE = '''Act as an independent adversarial judge. Return ONLY JSON with keys current_judgement, confidence, supported_claims, weak_or_unsupported_claims, contradictions, strongest_alternative, what_would_change_the_judgement, scores.
Score these 7 dimensions from 0-2: problem_definition, connection_quality, root_cause_quality, test_quality, evidence_discipline, uncertainty_honesty, challenge_resistance.
Do not reward confidence without evidence.
PROBLEM:\n{problem}\nCANDIDATES:\n{candidates}\nCRITIQUES:\n{critiques}\nEVIDENCE:\n{evidence}\nCHALLENGES:\n{challenges}'''

class Hypothesis(BaseModel):
    id: str
    statement: str
    supporting_observations: list[str] = []
    missing_evidence: list[str] = []
    falsifiers: list[str] = []
    prior_plausibility: float = Field(0.5, ge=0, le=1)

class Candidate(BaseModel):
    provider: str
    model: str
    observations: list[str] = []
    connections: list[str] = []
    hypotheses: list[Hypothesis] = []
    proposed_tests: list[str] = []
    evidence_requirements: list[str] = []
    uncertainties: list[str] = []
    error: str | None = None

class Challenge(BaseModel):
    challenger: str = 'public'
    claim_under_attack: str
    counterargument: str
    evidence: list[str] = []
    requested_test: str | None = None

class Evidence(BaseModel):
    id: str
    kind: str
    statement: str
    source: str
    timestamp: str
    status: str = 'unverified'

class Investigation(BaseModel):
    run_id: str
    created_at: str
    problem: str
    candidates: list[Candidate]
    merged_hypotheses: list[Hypothesis]
    tests: list[str]
    evidence_requirements: list[str]
    evidence: list[Evidence]
    challenges: list[Challenge] = []
    critiques: list[dict[str, Any]] = []
    judge: dict[str, Any]
    audit_log: list[str] = []

RUNS: dict[str, Investigation] = {}
app = FastAPI(title='MMI-PSAI X1', version='0.2.0')

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request as FastAPIRequest

limiter = Limiter(key_func=get_remote_address, default_limits=[os.getenv('MMI_RATE_LIMIT', '20/hour')])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

def now():
    return datetime.now(timezone.utc).isoformat()

def json_clean(text: str):
    t = (text or '').strip()
    if t.startswith('```'):
        t = t.split('\n', 1)[1] if '\n' in t else t
        t = t.rsplit('```', 1)[0]
    try:
        return json.loads(t)
    except Exception:
        a, b = t.find('{'), t.rfind('}')
        if a >= 0 and b > a:
            return json.loads(t[a:b+1])
        raise ValueError('Provider did not return valid JSON')

def mock_candidate(problem: str) -> Candidate:
    return Candidate(provider='mock', model='deterministic', observations=['The problem statement is not itself proof of a root cause.'], connections=['user context','system behaviour','environment/configuration'], hypotheses=[Hypothesis(id='H1', statement='Insufficient observation or missing context', missing_evidence=['raw logs','controlled reproduction'], falsifiers=['complete evidence excludes this']), Hypothesis(id='H2', statement='A system or configuration factor causes the behaviour', missing_evidence=['system metrics','configuration'], falsifiers=['controlled reproduction rules this out'])], proposed_tests=['Fresh-session reproduction','Change one variable at a time','Collect raw logs and independent measurements'], evidence_requirements=['reproduction record','logs','independent observation'], uncertainties=['Mock mode has no external evidence'])

async def openai_call(prompt: str):
    from openai import AsyncOpenAI
    key=os.getenv('OPENAI_API_KEY'); model=os.getenv('OPENAI_MODEL')
    if not key or not model: raise RuntimeError('OpenAI key/model not configured')
    client=AsyncOpenAI(api_key=key)
    r=await client.responses.create(model=model, instructions=CORE, input=prompt)
    return r.output_text, model

async def anthropic_call(prompt: str):
    from anthropic import AsyncAnthropic
    key=os.getenv('ANTHROPIC_API_KEY'); model=os.getenv('ANTHROPIC_MODEL')
    if not key or not model: raise RuntimeError('Anthropic key/model not configured')
    client=AsyncAnthropic(api_key=key)
    r=await client.messages.create(model=model, max_tokens=5000, system=CORE, messages=[{'role':'user','content':prompt}])
    return ''.join(getattr(x,'text','') for x in r.content), model

async def gemini_call(prompt: str):
    from google import genai
    key=os.getenv('GEMINI_API_KEY'); model=os.getenv('GEMINI_MODEL')
    if not key or not model: raise RuntimeError('Gemini key/model not configured')
    client=genai.Client(api_key=key)
    r=await asyncio.to_thread(client.models.generate_content, model=model, contents=CORE+'\n\n'+prompt)
    return r.text, model

async def ollama_call(prompt: str):
    base=os.getenv('OLLAMA_BASE_URL','http://localhost:11434').rstrip('/')
    model=os.getenv('OLLAMA_MODEL','llama3.1')
    async with httpx.AsyncClient(timeout=180) as c:
        r=await c.post(base+'/api/chat', json={'model':model,'stream':False,'format':'json','messages':[{'role':'system','content':CORE},{'role':'user','content':prompt}]})
        r.raise_for_status()
        return r.json()['message']['content'], model

async def call_provider(name: str, prompt: str):
    name=name.lower()
    if name=='mock': return json.dumps(mock_candidate('x').model_dump()), 'deterministic'
    if name=='openai': return await openai_call(prompt)
    if name=='anthropic': return await anthropic_call(prompt)
    if name=='gemini': return await gemini_call(prompt)
    if name=='ollama': return await ollama_call(prompt)
    raise RuntimeError('Unknown provider: '+name)

async def candidate_for(provider: str, problem: str):
    try:
        if provider=='mock':
            c=mock_candidate(problem); return c
        text, model=await asyncio.wait_for(call_provider(provider, ANALYZE.format(problem=problem)), 180)
        d=json_clean(text)
        return Candidate(provider=provider, model=model, **d)
    except Exception as e:
        return Candidate(provider=provider, model='error', error=str(e))

async def critique_for(provider: str, problem: str, cand: Candidate, others: list[Candidate]):
    if provider=='mock':
        return {'provider':'mock','attacks':['The conclusion may be over-generalized.'],'unsupported_claims':[],'missing_tests':['fresh-session reproduction'],'best_counter_hypothesis':'context or external configuration','additional_evidence_needed':['raw logs','controlled comparison']}
    try:
        text,_=await call_provider(provider, CRITIQUE.format(problem=problem,candidate=cand.model_dump_json(),others='\n'.join(x.model_dump_json() for x in others)))
        d=json_clean(text); d['provider']=provider; return d
    except Exception as e:
        return {'provider':provider,'error':str(e)}

async def judge_for(provider: str, prompt: str):
    try:
        if provider=='mock':
            return {'current_judgement':'No unique root cause is established from supplied evidence.','confidence':0.35,'supported_claims':['Multiple hypotheses can be stated without overclaiming.'],'weak_or_unsupported_claims':['Any hidden mechanism claimed as fact.'],'contradictions':[],'strongest_alternative':'Missing context or an external system factor.','what_would_change_the_judgement':['controlled reproduction','independent evidence'],'scores':{'problem_definition':2,'connection_quality':2,'root_cause_quality':1,'test_quality':2,'evidence_discipline':2,'uncertainty_honesty':2,'challenge_resistance':1}}
        text,_=await call_provider(provider,prompt); return json_clean(text)
    except Exception:
        return await judge_for('mock',prompt)

async def investigate(problem: str):
    names=[x.strip().lower() for x in os.getenv('MMI_PROVIDERS','mock').split(',') if x.strip()]
    if not names: names=['mock']
    sem=asyncio.Semaphore(int(os.getenv('MMI_MAX_PARALLEL','4')))
    async def guarded(p):
        async with sem: return await candidate_for(p,problem)
    candidates=await asyncio.gather(*(guarded(p) for p in names))
    candidates=[c for c in candidates if c.error is None] or [mock_candidate(problem)]
    merged=[]; seen=set()
    for c in candidates:
        for h in c.hypotheses:
            k=h.statement.lower().strip()
            if k not in seen: seen.add(k); merged.append(h)
    tests=sorted({t for c in candidates for t in c.proposed_tests})
    reqs=sorted({e for c in candidates for e in c.evidence_requirements})
    evidence=[Evidence(id=f'OBS-{i+1}',kind='model',statement=o,source=f'{c.provider}:{c.model}',timestamp=now()) for i,c in enumerate(candidates) for o in c.observations]
    critiques=[]
    for p in names:
        for c in candidates:
            critiques.append(await critique_for(p,problem,c,candidates))
    prompt=JUDGE.format(problem=problem,candidates='\n'.join(c.model_dump_json() for c in candidates),critiques=json.dumps(critiques,ensure_ascii=False),evidence='\n'.join(e.model_dump_json() for e in evidence),challenges='[]')
    judge=await judge_for(names[0],prompt)
    inv=Investigation(run_id=str(uuid.uuid4()),created_at=now(),problem=problem,candidates=candidates,merged_hypotheses=merged,tests=tests,evidence_requirements=reqs,evidence=evidence,critiques=critiques,judge=judge,audit_log=[f'providers={names}',f'candidates={len(candidates)}',f'hypotheses={len(merged)}','model outputs are unverified until independently supported'])
    RUNS[inv.run_id]=inv
    return inv

@app.get('/', response_class=HTMLResponse)
def home():
    return HTML

@app.get('/health')
def health(): return {'ok':True,'system':'MMI-PSAI X1','version':'0.2.0'}

class Request(BaseModel):
    problem: str = Field(min_length=3,max_length=30000)

@app.post('/v1/investigate')
@limiter.limit(os.getenv('MMI_INVESTIGATE_LIMIT', '5/minute'))
async def api_investigate(request: FastAPIRequest, req: Request): return await investigate(req.problem)

@app.post('/v1/challenge/{run_id}')
@limiter.limit(os.getenv('MMI_INVESTIGATE_LIMIT', '5/minute'))
async def api_challenge(request: FastAPIRequest, run_id: str, challenge: Challenge):
    inv=RUNS.get(run_id)
    if not inv: raise HTTPException(404,'run not found')
    inv.challenges.append(challenge)
    inv.evidence.append(Evidence(id=f'CH-{len(inv.challenges)}',kind='public_challenge',statement=challenge.counterargument,source=challenge.challenger,timestamp=now(),status='unverified'))
    prompt=JUDGE.format(problem=inv.problem,candidates='\n'.join(c.model_dump_json() for c in inv.candidates),critiques=json.dumps(inv.critiques,ensure_ascii=False),evidence='\n'.join(e.model_dump_json() for e in inv.evidence),challenges='\n'.join(c.model_dump_json() for c in inv.challenges))
    inv.judge=await judge_for(inv.candidates[0].provider,prompt)
    inv.audit_log.append('public challenge received and judge re-run')
    return inv

@app.get('/v1/run/{run_id}')
def api_run(run_id: str):
    if run_id not in RUNS: raise HTTPException(404,'run not found')
    return RUNS[run_id]

HTML='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MMI-PSAI X1</title><style>body{font-family:system-ui;max-width:1000px;margin:40px auto;padding:0 16px;line-height:1.5}textarea{width:100%;min-height:160px;box-sizing:border-box;padding:12px;font-size:16px}button{padding:10px 16px;margin:8px 4px 8px 0}.card{border:1px solid #ddd;border-radius:12px;padding:16px;margin:16px 0}pre{white-space:pre-wrap;word-break:break-word;background:#f6f6f6;padding:12px;border-radius:8px}</style></head><body><h1>MMI-PSAI X1</h1><p>Problem = Solution Intelligence System</p><p>Experimental research prototype: results are attackable and should be challenged.</p><textarea id="p" placeholder="Describe the problem..."></textarea><br><button onclick="run()">Investigate</button><div id="o"></div><script>let id='';function esc(s){return String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;')}async function run(){o.innerHTML='<div class="card">Investigating...</div>';let r=await fetch('/v1/investigate',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({problem:p.value})});let d=await r.json();id=d.run_id;render(d)}async function challenge(){let a=prompt('Claim to attack?');let b=prompt('Counterargument/evidence?');if(!a||!b)return;let r=await fetch('/v1/challenge/'+id,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({claim_under_attack:a,counterargument:b,evidence:[],challenger:'public-user'})});render(await r.json())}function render(d){o.innerHTML='<div class="card"><h2>Current judgement</h2><p>'+esc(d.judge.current_judgement)+'</p><p>Confidence: '+Math.round(d.judge.confidence*100)+'%</p><button onclick="challenge()">Challenge result</button></div><div class="card"><h3>Root hypotheses</h3><pre>'+esc(JSON.stringify(d.merged_hypotheses,null,2))+'</pre></div><div class="card"><h3>Tests</h3><pre>'+esc(JSON.stringify(d.tests,null,2))+'</pre></div><div class="card"><h3>Evidence ledger</h3><pre>'+esc(JSON.stringify(d.evidence,null,2))+'</pre></div></div>'}</script></body></html>'''
