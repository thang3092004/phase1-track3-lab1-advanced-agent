# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_100.json
- Mode: openai
- Records: 200
- Agents: react, reflexion

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.46 | 0.59 | 0.13 |
| Avg attempts | 1 | 2 | 1 |
| Avg token estimate | 1772.28 | 5405.29 | 3633.01 |
| Avg latency (ms) | 2001.97 | 5140 | 3138.03 |

## Failure modes
```json
{
  "react": {
    "none": 46,
    "wrong_final_answer": 54
  },
  "reflexion": {
    "none": 59,
    "wrong_final_answer": 41
  }
}
```

## Extensions implemented
- structured_evaluator
- reflection_memory
- benchmark_report_json

## Discussion
Reflexion helps when the first attempt stops after the first hop or drifts to a wrong second-hop entity. The tradeoff is higher attempts, token cost, and latency. In a real report, students should explain when the reflection memory was useful, which failure modes remained, and whether evaluator quality limited gains.
