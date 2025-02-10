import json
import re
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np

from grading.grader import grade_answer
from models import get_messages_with_few_shot_prompt


@dataclass
class MATHQuestion:
    problem: str
    answer: str
    solution: str
    subject: str
    level: int
    unique_id: str

    def get_prompt(self) -> str:
        return f"{self.problem}\n\nPlease enclose your final answer in <answer></answer> tags."

    @staticmethod
    def parse_response_for_answer(response: str) -> str:
        answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if answer_match is None:
            return ""
        return answer_match.group(1)


def load_questions(
    split: Literal["train", "test"],
    num_examples: int | None = None,
    subject: str | None = None,
    level: int | None = None,
) -> list[MATHQuestion]:
    """Load questions from a JSONL file."""
    np.random.seed(42)
    with open(f"math_splits/{split}.jsonl") as f:
        raw_data = [json.loads(line) for line in f]
    all_questions = [MATHQuestion(**d) for d in raw_data]
    if subject is not None:
        all_questions = [q for q in all_questions if q.subject == subject]
    if level is not None:
        all_questions = [q for q in all_questions if q.level == level]

    if num_examples is not None and num_examples < len(all_questions):
        idxs = np.random.choice(len(all_questions), size=num_examples, replace=False)
        return [all_questions[i] for i in idxs]
    return all_questions


def grade_question(question: MATHQuestion, model_answer: str) -> bool:
    return grade_answer(model_answer, question.answer)


def eval_model_answers(dataset: list[MATHQuestion], model_answers: list[str]) -> list[bool]:
    """Evaluate a list of model answers against their corresponding questions."""
    return [
        grade_question(question, answer)
        for question, answer in zip(dataset, model_answers, strict=True)
    ]


async def evaluate_model(
    dataset: list[MATHQuestion], model: str, few_shot_prompts: list[list[dict]] = []
) -> tuple[float, list[bool]]:
    """Evaluate model performance on the given dataset."""
    prompts = [q.get_prompt() for q in dataset]
    responses = await get_messages_with_few_shot_prompt(
        few_shot_prompts=few_shot_prompts, prompts=prompts, model=model
    )
    model_answers = [MATHQuestion.parse_response_for_answer(r.content[0].text) for r in responses]
    results = eval_model_answers(dataset, model_answers)
    accuracy = np.mean(results)
    return accuracy, results


if __name__ == "__main__":
    dataset = load_questions("train", subject="Precalculus")
    print(f"Precalculus: {len(dataset)}")
    dataset = load_questions("train", level=5)
    print(f"Level 5: {len(dataset)}")
    dataset = load_questions("train", subject="Precalculus", level=5)
    print(f"Precalculus & Level 5: {len(dataset)}")

    test_dataset = load_questions("test")
    print(f"Test: {len(test_dataset)}")
