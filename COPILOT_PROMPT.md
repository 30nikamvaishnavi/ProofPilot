# Prompt for GitHub Copilot (VS Code, Agent mode)

Open this folder in VS Code, open Copilot Chat in **Agent** mode, and paste everything under the line.

---

You are finishing and hardening the Python project **ProofPilot** in this workspace (a verified multi-step STEM agent:
plan → solve → verify → repair → finalize). Read `README.md`, `src/proofpilot/engine.py`, `graph.py`, `api.py` and `tools/base.py` first.

## Current state (important)
* The core logic (tools + verifiers, planner, memory, engine, spec module, eval) was written and tested in an environment WITHOUT
  pydantic, pytest, fastapi, langgraph or hypothesis installed. 223 test cases passed there using stand-ins for pydantic/pytest.
* These were NEVER executed against the real libraries: `tests/test_api.py`, `tests/test_graph.py`, `tests/test_properties.py`,
  `src/proofpilot/graph.py`, `src/proofpilot/api.py`, `src/proofpilot/llm.py` (Gemini SDK), and the real pydantic v2 behaviour of `schemas.py` and tool `ArgsModel`s.
  Expect small API mismatches there. Fix them in the adapters, not by weakening tests.

## Tasks (do them in order; run the commands yourself and show the results)
1. `python -m venv .venv`, activate, `pip install -r requirements-dev.txt && pip install -e .`
2. Run `pytest -q`. Fix every failure. Likely areas: pydantic v2 validation differences (strictness of `bool`/`int`/`str`),
   LangGraph API details (`StateGraph`, `add_conditional_edges`, `SqliteSaver`, `get_state`), FastAPI TestClient, Hypothesis health checks.
   Rules: do not delete or loosen tests; do not change verifier semantics; if a test expectation is truly wrong, explain why before editing.
3. Run `ruff check src tests eval` and `mypy src`; fix all findings (keep line-length 110).
4. Run `python -m eval.run_eval --fault-rate 0.3 --langgraph`. The custom and LangGraph rows must have identical accuracy.
   Commit the regenerated `eval/RESULTS.md` and paste the table into the README "Evaluation" section.
5. Verify the Gemini path: with `GEMINI_API_KEY` set, run `proofpilot "find the area under sin(x) between 0 and pi, then check if 91 is prime" --trace`.
   If the `google-genai` call in `llm.py` fails, fix it against the current SDK docs (keep `complete_json(prompt) -> dict`, temperature 0, JSON output).
   Offline behaviour (no key) must stay unchanged.
6. Start the API (`proofpilot serve`) and call `/health`, `/solve`, and the `/spec/*` flow with curl. Then `docker build` and run the container.
7. Add a short "Orchestrator comparison" section to the README: lines of code of `engine.py` loop vs `graph.py`, behaviour parity (from step 4), what LangGraph adds (checkpoint/resume), what it costs.
8. Add a minimal Streamlit UI `app_ui.py` (text box → report with step table, trace and verification methods; second tab for Paper-to-Pipeline with an explicit Approve button before code is shown). Add `streamlit` to an optional extra, not to core requirements.
9. Optional: replace the API-level spec approval with a LangGraph `interrupt` human-in-the-loop node while keeping the existing endpoints working and tested.
10. Deploy to Render or Hugging Face Spaces using the Dockerfile / `render.yaml`; put the live URL in the README and in `/health` smoke test in CI.

## Constraints
* Keep the architecture: every `Tool` has `run` + an independent `verify`; verified results only go to memory; engine and graph share the same node functions.
* Never call `eval`/`exec`/`sympify` on user text. Keep `safe_parse` and `spec.parse_safe` protections.
* Type hints everywhere, Python >= 3.10, no new heavy dependencies without justification.
* After each task, summarise what changed and what you ran. Do not claim something works unless you ran it.
