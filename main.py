# %%
import asyncio
import json
import os
import random
import re
from dataclasses import dataclass
from time import time
from typing import Literal

import nest_asyncio
import numpy as np
from anthropic import AI_PROMPT, HUMAN_PROMPT, AsyncAnthropic, RateLimitError
from anthropic.types import Message
from dotenv import load_dotenv

from grading.grader import grade_answer

load_dotenv()
nest_asyncio.apply()  # Allows nested event loops in Jupyter
anthropic = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY_TAKEHOME"])
model_nicknames = {
    "sonnet-35": "claude-3-5-sonnet-20240620",
    "opus-3": "claude-3-opus-20240229",
    "sonnet-3": "claude-3-sonnet-20240229",
    "haiku-3": "claude-3-haiku-20240307",
}


def get_few_shot_prompt(prompts_and_responses: list[tuple[str, str]]) -> list[dict]:
    """
    Formats a set of few-shot examples into something ingestible by the anthropic api client.

    Args:
      prompts_and_responses: A list of paired prompts and responses -- the prompts and separators are assumed to not contain the human and assistant separators.
    """
    messages = []
    for p, r in prompts_and_responses:
        assert HUMAN_PROMPT not in p, "No need to place the human separator in the prompts!"
        assert AI_PROMPT not in r, "No need to place the assistant separator in the responses!"
        messages.append({"role": "user", "content": [{"type": "text", "text": p}]})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": r}]})

    return messages


few_shot_prompt = get_few_shot_prompt(
    [("What is 2 + 2?", "2 + 2 = 4."), ("What is 49*7?", "49 * 7 = 343.")]
)
print("Few Shot Prompt Messages:")
for message in few_shot_prompt:
    role = message["role"].capitalize()
    text = message["content"][0]["text"]
    print(f"{role}: {text}")

# %%
MAX_PARALLEL_REQUESTS = 20
semaphore = asyncio.Semaphore(MAX_PARALLEL_REQUESTS)


# Example of getting a response with a few-shot prompt prepended
async def get_message_with_few_shot_prompt(
    few_shot_prompt: list[Message], prompt: str, model: str = "sonnet-35", max_retries: int = 5
) -> Message:
    messages = few_shot_prompt + [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    if model in model_nicknames:
        model = model_nicknames[model]

    async with semaphore:
        for attempt in range(max_retries):
            try:
                start = time()
                message = await anthropic.with_options(max_retries=max_retries).messages.create(
                    model=model, max_tokens=1000, temperature=0, messages=messages
                )
                print(f"Got response from {model} after {time() - start:.2f}s")
                return message
            except RateLimitError as e:
                if attempt == max_retries - 1:
                    raise  # Re-raise the exception if we've exhausted all retries

                wait_time = (2**attempt) + random.random()
                print(f"Rate limit error: {e}. Retrying in {wait_time:.2f} seconds...")
                await asyncio.sleep(wait_time)

    raise RuntimeError("Retries exhausted")  # Should not get here


message = asyncio.run(
    get_message_with_few_shot_prompt(few_shot_prompt, prompt="What is 64 ** 2?", model="haiku-3")
)
print(f"Final message content:\n{message.content}")
print(f"Final text response:\n{message.content[0].text}")


# %%
# Example of getting a list of responses to prompts with a few-shot prompt prepended
async def get_messages_with_few_shot_prompt(
    few_shot_prompt: list[Message],
    prompts: list[str],
    model: str = "sonnet-35",
    max_retries: int = 5,
) -> list[Message]:
    messages = await asyncio.gather(
        *[
            get_message_with_few_shot_prompt(
                few_shot_prompt, prompt=p, model=model, max_retries=max_retries
            )
            for p in prompts
        ]
    )
    return messages


messages = asyncio.run(
    get_messages_with_few_shot_prompt(
        few_shot_prompt, ["What is 64 ** 2?", "What is 243 / 7?", "What is 999*8?"], model="haiku-3"
    )
)
# Print each message's text content in a readable format
for i, msg in enumerate(messages, 1):
    print(f"\nMessage {i}:")
    print(msg.content[0].text)

# %%


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


def load_questions(split: Literal["train", "test"]) -> list[MATHQuestion]:
    with open(f"math_splits/{split}.jsonl") as f:
        raw_data = [json.loads(line) for line in f]
    return [MATHQuestion(**d) for d in raw_data]


def grade_question(question: MATHQuestion, model_answer: str) -> bool:
    return grade_answer(model_answer, question.answer)


def eval_model_answers(dataset: list[MATHQuestion], model_answers: list[str]) -> list[bool]:
    "Simple convenience function to evaluate a list of model answers  to a list of questions. Note that the answer must first be extracted from the response before being passed in here."
    return [
        grade_question(question, answer)
        for question, answer in zip(dataset, model_answers, strict=True)
    ]


train_dataset = load_questions("train")
test_dataset = load_questions("test")

print(test_dataset[0].get_prompt())

np.mean(eval_model_answers(train_dataset, [q.answer for q in train_dataset]))

# %%
