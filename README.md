# ProofPilot: a verified, multi-step STEM agent

ProofPilot turns a STEM problem into a plan, runs each step with a tool, **checks every answer with an independent
second method**, retries or repairs failures, and returns a typed, traceable report.

```
problem ──► plan ──► solve ──► verify ──┬─► commit ──► (next step) ──► finalize ──► Report
              │        ▲                │
              │        └──── repair ◄───┘  (verifier disagreed or tool crashed; max 3 attempts)
              └─ LLM planner (Gemini) if GEMINI_API_KEY is set, rule-based planner otherwise
```

* **Planner**: LLM output is schema-checked (tool exists, arguments validate) and falls back to rules on any problem. It works with no API key.
* **Tools (11)**: *math* (solve, differentiate, integrate, matrix ops, linear systems, number theory, SymPy), *CS* (shortest path, MST, topological sort, NetworkX), *finance* (Black-Scholes, portfolio variance).
* **Verifiers**: every tool ships a different method for checking its own answer, for example back-substitution + `numpy.roots`, finite differences, `scipy.quad`, Bellman-Ford + brute force, Monte Carlo + put-call parity, Miller-Rabin.
* **Repair**: execution errors get argument clean-up then LLM repair. Verifier disagreements get a recompute, then LLM repair.
* **Memory**: verified results are stored in SQLite and reused in later steps and runs. Unverified results are never stored.
* **State and structure**: a typed LangGraph state, Pydantic schemas for every payload, a plugin-style tool registry (ABC).
* **Two orchestrators, one set of nodes**: `engine.py` (custom loop) and `graph.py` (LangGraph with SQLite checkpoints, resumable by `thread_id`).
* **Paper-to-Pipeline**: theorem text → JSON spec → **human approval** → randomised counter-example search → generated Hypothesis test module.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt && pip install -e .
pytest -q
proofpilot "solve x^2-5x+6=0; then is 97 prime" --trace
proofpilot spec "For all integers a, b: a*b = b*a" --approve
proofpilot serve            # http://127.0.0.1:8000/docs
python -m eval.run_eval --fault-rate 0.3 --langgraph
```

Optional: copy `.env.example` to `.env` and set `GEMINI_API_KEY` for LLM planning/repair (`GEMINI_MODEL` is configurable).

## Example problems

```
solve x^2 - 5x + 6 = 0
integrate sin(x) from 0 to pi
differentiate x^2*sin(x)
determinant of [[1,2],[3,4]]        eigenvalues of [[2,0],[0,3]]
solve A=[[2,1],[1,3]] b=[3,5]
is 7919 prime        factorize 360        gcd of 84 and 36
shortest path from A to D in [['A','B',4],['A','C',1],['C','B',2],['B','D',5]]
minimum spanning tree of [[...]]        topological order of [['a','b'],['b','c']]
price a call S=100 K=105 T=1 r=0.05 sigma=0.2
portfolio variance weights=[0.5,0.5] cov=[[0.04,0.01],[0.01,0.09]]
```
Chain steps with `;` or `then`. Free-form phrasing needs the LLM planner.

## HTTP API

`GET /health` · `GET /tools` · `POST /solve {problem, plan?, orchestrator: custom|langgraph}` · `GET /memory`
`POST /spec/draft` → `POST /spec/{id}/approve` → `GET /spec/{id}/check` · `GET /spec/{id}/code` (409 until approved).
Set `PROOFPILOT_API_KEY` to require an `X-API-Key` header.

## Evaluation (64 problems, hand-checked answers)

Measured by `python -m eval.run_eval --fault-rate 0.3 --seed 0` (custom orchestrator, no network, no LLM).
"Injected faults" corrupt tool outputs at random to simulate an unreliable solver.

| orchestrator | condition | verifier | accuracy | verified | wrong but marked verified | mean attempts | retried | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|
| custom | no faults | on | 100.0% | 100.0% | 0 | 1.00 | 0 | 5.4 | 52.7 |
| custom | no faults | off | 100.0% | 0.0% | 0 | 1.00 | 0 | 2.2 | 11.9 |
| custom | 30% injected faults | on | 98.4% | 98.4% | 0 | 1.22 | 13 | 7.6 | 32.5 |
| custom | 30% injected faults | off | 84.4% | 0.0% | 0 | 1.00 | 0 | 2.5 | 15.0 |
| langgraph | no faults | on | 100.0% | 100.0% | 0 | 1.00 | 0 | 12.4 | 29.4 |
| langgraph | no faults | off | 100.0% | 0.0% | 0 | 1.00 | 0 | 7.3 | 22.4 |
| langgraph | 30% injected faults | on | 98.4% | 98.4% | 0 | 1.22 | 13 | 14.5 | 38.8 |
| langgraph | 30% injected faults | off | 84.4% | 0.0% | 0 | 1.00 | 0 | 6.0 | 17.8 |

Reading it:
* With no faults every tool is already correct, so the verifier adds no accuracy and costs about 2-3 ms per problem.
* With 30% corrupted outputs, accuracy is 98.4% with the verifier and 84.4% without (seed 7: 95.3% vs 70.3%).
* **No wrong answer was ever marked verified.** The remaining misses are problems where all 3 attempts were corrupted. They are reported as *unverified*, not as wrong answers.
* This measures the verify/repair loop against simulated faults. It is not a benchmark of LLM math ability.

## Orchestrator comparison

The custom loop in `engine.py` is the smaller orchestration: roughly 240 lines for state, nodes, routing, and the loop.
The LangGraph adapter in `graph.py` is roughly 80 lines because it reuses the same node and router functions.
On the 64-problem, 30%-fault run above, custom and LangGraph both achieve 98.4% accuracy, and both retain 98.4%
verified accuracy with no wrong answer marked verified. LangGraph adds SQLite checkpoint persistence and resume through
`thread_id`, plus graph lifecycle/state tooling. Its cost is extra framework/import overhead (observed p50/p95 latency
was higher on this run) and checkpoint serialization/storage work.

## What is tested

`pytest` covers every tool, every verifier (including a mutation test that corrupts each tool's answer and requires rejection), the planner, repair paths, timeouts, memory, the spec workflow and its safety checks, the API, LangGraph parity with the custom engine, and property tests.

## Security notes

* Math input is never passed to `sympify`/`eval` directly: character whitelist, identifier whitelist, attribute-access ban, and DoS guards (power towers, exponents > 100, 19+ digit numbers).
* Theorem claims are parsed with `ast`, checked against an allow-list, and run by a small interpreter, never `eval`.
* Tool calls have timeouts. LLM text is treated as data: plans and repairs are schema-validated.
* This is not a hardened sandbox. For untrusted public traffic add rate limiting and process isolation.

## Limitations

* The rule-based planner understands the phrasings above only. Use an LLM key for free-form problems.
* Verification is evidence from an independent method, not a formal proof. Randomised spec checks find counter-examples; they do not prove theorems.
* Matrices are limited to 8x8, graphs to 500 edges, integers to 10^18.
* Black-Scholes assumes constant volatility, no dividends, European exercise.

## Deploy

`docker build -t proofpilot . && docker run -p 8000:8000 proofpilot`, or connect the repo to Render (`render.yaml` included). Put the live URL here once deployed: **TBD**.

## Layout

```
src/proofpilot/  schemas.py  engine.py (nodes + custom loop)  graph.py (LangGraph)  planner.py  llm.py
                 memory.py  spec.py  faults.py  api.py  cli.py  tools/{base,math_tools,cs_tools,finance_tools}.py
tests/  eval/  Dockerfile  render.yaml  .github/workflows/ci.yml  REVIEW_LOG.md  COPILOT_PROMPT.md
```
