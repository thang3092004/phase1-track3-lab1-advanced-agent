from __future__ import annotations

import json
import os
from dataclasses import dataclass
from time import perf_counter

import requests
from dotenv import load_dotenv

from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import JudgeResult, QAExample, ReflectionEntry
from .utils import normalize_answer


@dataclass
class LLMCall:
    text: str
    token_estimate: int
    latency_ms: int


class OpenAILabRuntime:
    def __init__(self, model: str = "gpt-3.5-turbo") -> None:
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is missing. Add it to your .env file.")

        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model = model
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

    @staticmethod
    def _strip_markdown_fence(text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```") and cleaned.endswith("```"):
            lines = cleaned.splitlines()
            if len(lines) >= 3:
                return "\n".join(lines[1:-1]).strip()
        return cleaned

    @staticmethod
    def _coerce_list(value: object) -> list[str]:
        if isinstance(value, list):
            return [str(item) for item in value]
        if value in (None, ""):
            return []
        return [str(value)]

    @staticmethod
    def _context_to_text(example: QAExample) -> str:
        return "\n".join(f"[{idx}] {chunk.title}: {chunk.text}" for idx, chunk in enumerate(example.context, start=1))

    def _chat(self, system_prompt: str, user_prompt: str, expect_json: bool = False) -> LLMCall:
        start = perf_counter()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
        }
        if expect_json:
            payload["response_format"] = {"type": "json_object"}

        response = self.session.post(f"{self.base_url}/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()

        latency_ms = int((perf_counter() - start) * 1000)

        content = ""
        if "choices" in data and data["choices"] and "message" in data["choices"][0] and "content" in data["choices"][0]["message"]:
            content = data["choices"][0]["message"]["content"].strip()

        total_tokens = 0
        if "usage" in data and "total_tokens" in data["usage"]:
            total_tokens = int(data["usage"]["total_tokens"])

        return LLMCall(text=content, token_estimate=total_tokens, latency_ms=latency_ms)

    def actor_answer(
        self,
        example: QAExample,
        attempt_id: int,
        agent_type: str,
        reflection_memory: list[str],
    ) -> LLMCall:
        reflection_block = "\n".join(f"- {item}" for item in reflection_memory) if reflection_memory else "(none)"
        user_prompt = (
            f"Question:\n{example.question}\n\n"
            f"Context:\n{self._context_to_text(example)}\n\n"
            f"Attempt: {attempt_id}\n"
            f"Agent type: {agent_type}\n"
            f"Reflection memory:\n{reflection_block}\n\n"
            "Return only the final answer string."
        )
        return self._chat(ACTOR_SYSTEM, user_prompt, expect_json=False)

    def evaluator(self, example: QAExample, answer: str) -> tuple[JudgeResult, LLMCall]:
        user_prompt = (
            f"Question:\n{example.question}\n\n"
            f"Gold answer:\n{example.gold_answer}\n\n"
            f"Predicted answer:\n{answer}\n\n"
            "Evaluate correctness and return JSON with keys: score, reason, missing_evidence, spurious_claims."
        )
        call = self._chat(EVALUATOR_SYSTEM, user_prompt, expect_json=True)

        try:
            payload = json.loads(self._strip_markdown_fence(call.text))
            score = int(payload.get("score", 0))
            payload["score"] = 1 if score == 1 else 0
            payload["reason"] = str(payload.get("reason", "No reason provided."))
            payload["missing_evidence"] = self._coerce_list(payload.get("missing_evidence"))
            payload["spurious_claims"] = self._coerce_list(payload.get("spurious_claims"))
            judge = JudgeResult.model_validate(payload)
        except Exception:
            is_correct = normalize_answer(example.gold_answer) == normalize_answer(answer)
            judge = JudgeResult(
                score=1 if is_correct else 0,
                reason="Evaluator JSON parsing failed; used fallback exact-match evaluator.",
                missing_evidence=[] if is_correct else ["Could not parse evaluator feedback."],
                spurious_claims=[] if is_correct else [answer],
            )

        return judge, call

    def reflector(
        self,
        example: QAExample,
        attempt_id: int,
        answer: str,
        judge: JudgeResult,
    ) -> tuple[ReflectionEntry, LLMCall]:
        user_prompt = (
            f"Question:\n{example.question}\n\n"
            f"Context:\n{self._context_to_text(example)}\n\n"
            f"Attempt id: {attempt_id}\n"
            f"Predicted answer: {answer}\n"
            f"Judge reason: {judge.reason}\n"
            f"Missing evidence: {judge.missing_evidence}\n"
            f"Spurious claims: {judge.spurious_claims}\n\n"
            "Return JSON with keys: failure_reason, lesson, next_strategy."
        )
        call = self._chat(REFLECTOR_SYSTEM, user_prompt, expect_json=True)

        try:
            payload = json.loads(self._strip_markdown_fence(call.text))
            payload["attempt_id"] = attempt_id
            payload["failure_reason"] = str(payload.get("failure_reason", judge.reason))
            payload["lesson"] = str(payload.get("lesson", "The final answer must match evidence in context."))
            payload["next_strategy"] = str(payload.get("next_strategy", "Re-check each hop and verify the final entity before answering."))
            reflection = ReflectionEntry.model_validate(payload)
        except Exception:
            reflection = ReflectionEntry(
                attempt_id=attempt_id,
                failure_reason=judge.reason,
                lesson="Previous attempt did not satisfy evaluator criteria.",
                next_strategy="Trace all hops explicitly and verify the final entity against context before answering.",
            )

        return reflection, call
