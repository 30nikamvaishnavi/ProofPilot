# Evaluation results (64 problems, seed 0)

| orchestrator | condition | verifier | accuracy | verified | wrong but marked verified | mean attempts | retried | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|
| custom | no faults | on | 100.0% | 100.0% | 0 | 1.00 | 0 | 7.8 | 77.7 |
| custom | no faults | off | 100.0% | 0.0% | 0 | 1.00 | 0 | 2.3 | 20.6 |
| custom | 30% injected faults | on | 98.4% | 98.4% | 0 | 1.22 | 13 | 7.6 | 32.5 |
| custom | 30% injected faults | off | 84.4% | 0.0% | 0 | 1.00 | 0 | 2.5 | 15.0 |
| langgraph | no faults | on | 100.0% | 100.0% | 0 | 1.00 | 0 | 12.4 | 29.4 |
| langgraph | no faults | off | 100.0% | 0.0% | 0 | 1.00 | 0 | 7.3 | 22.4 |
| langgraph | 30% injected faults | on | 98.4% | 98.4% | 0 | 1.22 | 13 | 14.5 | 38.8 |
| langgraph | 30% injected faults | off | 84.4% | 0.0% | 0 | 1.00 | 0 | 6.0 | 17.8 |

Faults are injected by `proofpilot.faults` (seeded). 'verifier off' accepts the first answer. Expected answers are in `eval/problems.py`.
