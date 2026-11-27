"""Minimal Streamlit interface for ProofPilot."""

from __future__ import annotations

import streamlit as st

from proofpilot.engine import Runtime, run_engine
from proofpilot.llm import get_llm
from proofpilot.spec import SpecWorkflow

st.set_page_title("ProofPilot")
st.title("ProofPilot")

solve_tab, paper_tab = st.tabs(["Solve", "Paper-to-Pipeline"])

with solve_tab:
    problem = st.text_area("Problem", "integrate sin(x) from 0 to pi")
    orchestrator = st.selectbox("Orchestrator", ["custom", "langgraph"])
    if st.button("Solve", type="primary"):
        runtime = Runtime(llm=get_llm())
        if orchestrator == "langgraph":
            from proofpilot.graph import GraphAgent

            report = GraphAgent(runtime).run(problem)
        else:
            report = run_engine(problem, runtime)

        st.subheader(report.status)
        st.write(report.final_answer)
        st.caption(f"Confidence: {report.confidence:.2f} | Planner: {report.planner}")

        if report.steps:
            st.subheader("Steps")
            st.dataframe(
                [
                    {
                        "step": step.step_id,
                        "tool": step.tool,
                        "value": step.value,
                        "verified": step.verified,
                        "method": step.method,
                        "attempts": step.attempts,
                    }
                    for step in report.steps
                ],
                use_container_width=True,
            )

        st.subheader("Trace")
        st.dataframe(
            [
                {"time_s": event.t, "node": event.node, "step": event.step_id, "message": event.message}
                for event in report.trace
            ],
            use_container_width=True,
        )

with paper_tab:
    text = st.text_area("Theorem text", "For all integers a, b: a*b = b*a")
    workflow = SpecWorkflow(get_llm())
    if "draft_spec" not in st.session_state:
        st.session_state.draft_spec = None

    if st.button("Draft spec"):
        try:
            st.session_state.draft_spec = workflow.draft(text)
        except Exception as exc:
            st.error(str(exc))

    spec = st.session_state.draft_spec
    if spec is not None:
        st.subheader("Draft")
        st.json(spec.model_dump())
        check = workflow.check(spec.id, n=1000, seed=0)
        st.subheader("Randomised check")
        st.json(check)

        if st.button("Approve"):
            approved = workflow.approve(spec.id)
            st.success("Approved")
            st.subheader("Generated code")
            st.code(workflow.generate(approved.id), language="python")