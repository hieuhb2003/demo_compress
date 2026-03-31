from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import List

from openai import OpenAI

from src.config import Settings
from src.models import AppState, JudgeReference, JudgeScore


JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator assessing AI responses based on a provided "
    "reference answer. Rate responses strictly on accuracy, completeness, and relevance."
)

JUDGE_USER_TEMPLATE = """\
Question: {question}

Reference Answer: {reference_answer}

Model Response: {response}

Rate the model response on a scale of 1-10:
- Accuracy: Is the information factually correct compared to the reference?
- Completeness: Does it cover the key points from the reference answer?
- Relevance: Does it directly address the question asked?

Respond ONLY with valid JSON: {{"score": <integer 1-10>, "reasoning": "<1-2 sentence explanation>"}}"""


def _call_judge(
    api_key: str,
    base_url: str,
    model: str,
    question: str,
    reference_answer: str,
    response: str,
) -> tuple[float, str]:
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)

    prompt = JUDGE_USER_TEMPLATE.format(
        question=question,
        reference_answer=reference_answer,
        response=response,
    )
    result = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    text = result.choices[0].message.content or ""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    parsed = json.loads(text)
    return float(parsed["score"]), str(parsed.get("reasoning", ""))


def run_judge(app_state: AppState, settings: Settings) -> List[JudgeScore]:
    if not app_state.judge_references:
        return []

    ref_by_turn = {ref.turn_index: ref for ref in app_state.judge_references}

    tasks = []
    for method_key, state in app_state.method_states.items():
        for turn in state.turns:
            ref = ref_by_turn.get(turn.turn_index)
            if ref is None:
                continue
            tasks.append((turn.turn_index, method_key, ref.question, ref.reference_answer, turn.assistant_message))

    if not tasks:
        return []

    scores: List[JudgeScore] = []

    def _judge_one(task):
        turn_index, method_key, question, reference_answer, response = task
        try:
            score, reasoning = _call_judge(
                settings.openai_api_key,
                settings.llm_judge_base_url,
                settings.llm_judge_model,
                question,
                reference_answer,
                response,
            )
        except Exception as exc:
            score, reasoning = 0.0, f"Judge error: {exc}"
        return JudgeScore(
            turn_index=turn_index,
            method_key=method_key,
            score=score,
            reasoning=reasoning,
        )

    with ThreadPoolExecutor(max_workers=10) as executor:
        scores = list(executor.map(_judge_one, tasks))

    return scores


def load_judge_references(raw_json: list[dict], start_turn: int = 1) -> List[JudgeReference]:
    refs = []
    for i, item in enumerate(raw_json):
        question = item.get("question", "")
        reference_answer = item.get("reference_answer", "")
        if question:
            refs.append(JudgeReference(
                turn_index=start_turn + i,
                question=question,
                reference_answer=reference_answer,
            ))
    return refs
