from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Literal
from .mock_runtime import FAILURE_MODE_BY_QID, actor_answer, evaluator, reflector
from .schemas import AttemptTrace, JudgeResult, QAExample, ReflectionEntry, RunRecord
from .utils import normalize_answer

@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1
    mode: Literal["mock", "openai"] = "mock"
    model_name: str = "gpt-3.5-turbo"
    _runtime: Any = None

    def _get_runtime(self) -> Any:
        if self._runtime is None:
            from .llm_runtime import OpenAILabRuntime

            self._runtime = OpenAILabRuntime(model=self.model_name)
        return self._runtime

    @staticmethod
    def _fallback_failure_mode(example: QAExample, answer: str, judge: JudgeResult) -> str:
        reason = judge.reason.lower()
        if "incomplete" in reason or "multi-hop" in reason or "second hop" in reason:
            return "incomplete_multi_hop"
        if "entity" in reason or "drift" in reason:
            return "entity_drift"
        if normalize_answer(answer) == "":
            return "looping"
        return FAILURE_MODE_BY_QID.get(example.qid, "wrong_final_answer")

    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        final_judge: JudgeResult | None = None
        runtime = self._get_runtime() if self.mode == "openai" else None

        for attempt_id in range(1, self.max_attempts + 1):
            if self.mode == "openai" and runtime is not None:
                actor_call = runtime.actor_answer(example, attempt_id, self.agent_type, reflection_memory)
                answer = actor_call.text
                judge, evaluator_call = runtime.evaluator(example, answer)
                token_estimate = actor_call.token_estimate + evaluator_call.token_estimate
                latency_ms = actor_call.latency_ms + evaluator_call.latency_ms
            else:
                answer = actor_answer(example, attempt_id, self.agent_type, reflection_memory)
                judge = evaluator(example, answer)
                token_estimate = 320 + (attempt_id * 65) + (120 if self.agent_type == "reflexion" else 0)
                latency_ms = 160 + (attempt_id * 40) + (90 if self.agent_type == "reflexion" else 0)

            final_answer = answer
            final_score = judge.score
            final_judge = judge

            reflection_entry: ReflectionEntry | None = None
            if judge.score == 0 and self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                if self.mode == "openai" and runtime is not None:
                    reflection_entry, reflection_call = runtime.reflector(example, attempt_id, answer, judge)
                    token_estimate += reflection_call.token_estimate
                    latency_ms += reflection_call.latency_ms
                else:
                    reflection_entry = reflector(example, attempt_id, judge)

                reflections.append(reflection_entry)
                reflection_memory.append(
                    f"Attempt {attempt_id} failed. Lesson: {reflection_entry.lesson} Next strategy: {reflection_entry.next_strategy}"
                )

            trace = AttemptTrace(
                attempt_id=attempt_id,
                answer=answer,
                score=judge.score,
                reason=judge.reason,
                reflection=reflection_entry,
                token_estimate=token_estimate,
                latency_ms=latency_ms,
            )
            traces.append(trace)

            if judge.score == 1:
                break

        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        failure_mode = "none"
        if final_score != 1 and final_judge is not None:
            failure_mode = self._fallback_failure_mode(example, final_answer, final_judge)
        return RunRecord(qid=example.qid, question=example.question, gold_answer=example.gold_answer, agent_type=self.agent_type, predicted_answer=final_answer, is_correct=bool(final_score), attempts=len(traces), token_estimate=total_tokens, latency_ms=total_latency, failure_mode=failure_mode, reflections=reflections, traces=traces)

class ReActAgent(BaseAgent):
    def __init__(self, mode: Literal["mock", "openai"] = "mock", model_name: str = "gpt-4.1-mini") -> None:
        super().__init__(agent_type="react", max_attempts=1, mode=mode, model_name=model_name)

class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3, mode: Literal["mock", "openai"] = "mock", model_name: str = "gpt-4.1-mini") -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts, mode=mode, model_name=model_name)
