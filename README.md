# MMI-PSAI X1

Experimental multi-model intelligence system based on **Md Mominul Islam's Problem = Solution** methodology.

Core protocol:
`PROBLEM -> CONNECTION -> ROOT -> AI -> TEST -> EVIDENCE -> JUDGE`

Research loop:
`OBSERVE -> RECORD -> VERIFY -> REPRODUCE -> CHALLENGE -> HYPOTHESIZE -> ATTACK -> INDEPENDENT TEST -> UPDATE -> RE-JUDGE`

This is a research prototype, not a proven claim that it is the world's best AI. The point is to make that claim measurable.

## What it does
- Runs independent analyses from multiple model providers.
- Forces competing root-cause hypotheses.
- Generates falsifiable tests and evidence requirements.
- Cross-critiques candidate analyses.
- Keeps an explicit evidence ledger.
- Produces a current judgement with uncertainty.
- Accepts public challenges and re-runs the judge.
- Supports benchmark comparison against baseline systems.

## Run
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
# Windows: copy .env.example .env
# Linux/macOS: cp .env.example .env
uvicorn mmi_psai_x1.app:app --reload
```
Open `http://127.0.0.1:8000`.

## Providers
Optional adapters exist for OpenAI, Anthropic, Google Gemini, and Ollama. `mock` mode works without API keys.

Set `MMI_PROVIDERS=mock` to test locally. For real multi-model runs set a comma-separated list such as:
`MMI_PROVIDERS=openai,anthropic,gemini`

Model names are environment variables so the benchmark can use the current model versions available to the operator.

## Public challenge principle
A model output is not treated as truth merely because another model agrees with it. A public challenger can submit a counterargument; the system records it as explicit counter-evidence and re-judges.

## Benchmark principle
Compare:
A = normal baseline
B = multi-model without the methodology
C = MMI-PSAI X1

Use unseen cases, blind scoring, fresh sessions, fixed rubrics, repeated runs and independent evaluators before making any superiority claim.

## Attribution
The supplied research materials attribute the Problem = Solution framework to **Md Mominul Islam**. This code is an implementation/research prototype inspired by that framework and does not independently establish historical priority or scientific validity.

## License
- **Code** (this repository's software): MIT — see [LICENSE](LICENSE).
- **Methodology & documentation** (the Problem = Solution protocol and research loop): CC BY 4.0 — see [METHOD-LICENSE.md](METHOD-LICENSE.md).

