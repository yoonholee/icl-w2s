from anthropic import AI_PROMPT, HUMAN_PROMPT
from typing import Any


def get_few_shot_prompt(
    prompts_and_responses: list[tuple[str, str]], w2s_prompt: bool = False
) -> list[dict[str, Any]]:
    """Format few-shot examples for the Anthropic API."""
    messages = []
    for p, r in prompts_and_responses:
        assert HUMAN_PROMPT not in p, "No need to place the human separator in the prompts!"
        assert AI_PROMPT not in r, "No need to place the assistant separator in the responses!"
        if w2s_prompt and len(messages) == 0:
            p = (
                "The following are examples of math problem solutions from a weaker model. Use the format of the examples to write your own solution.\n\n"
                + p
            )
        messages.append({"role": "user", "content": [{"type": "text", "text": p}]})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": r}]})

    return messages


# Default few-shot examples
DEFAULT_FEW_SHOT = get_few_shot_prompt(
    [("What is 2 + 2?", "2 + 2 = 4."), ("What is 49*7?", "49 * 7 = 343.")]
)
