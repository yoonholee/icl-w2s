import asyncio
import os
import random
from time import time

from anthropic import AsyncAnthropic, RateLimitError
from anthropic.types import Message
from dotenv import load_dotenv
from tqdm.asyncio import tqdm
from diskcache import Cache
import hashlib
import json

load_dotenv()
anthropic = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY_TAKEHOME"])

model_nicknames = {
    "sonnet-35": "claude-3-5-sonnet-20240620",
    "opus-3": "claude-3-opus-20240229",
    "sonnet-3": "claude-3-sonnet-20240229",
    "haiku-3": "claude-3-haiku-20240307",
}

MAX_PARALLEL_REQUESTS = 20
semaphore = asyncio.Semaphore(MAX_PARALLEL_REQUESTS)

# Initialize cache with 1GB size limit in the .cache directory
cache = Cache(".cache", size_limit=1e9)


def get_cache_key(few_shot_prompt: list[Message], prompt: str, model: str) -> str:
    """Generate a deterministic cache key from the input parameters."""
    # Convert inputs to a stable string representation
    data = json.dumps(
        {"few_shot": few_shot_prompt, "prompt": prompt, "model": model}, sort_keys=True
    )
    return hashlib.sha256(data.encode()).hexdigest()


async def get_message_with_few_shot_prompt(
    few_shot_prompt: list[Message],
    prompt: str,
    model: str = "sonnet-35",
    max_retries: int = 5,
    silent: bool = True,
    use_cache: bool = True,
) -> Message:
    """Get a single model response with few-shot prompting."""
    if use_cache:
        cache_key = get_cache_key(few_shot_prompt, prompt, model)
        cached_response = cache.get(cache_key)
        if cached_response is not None:
            if not silent:
                print(f"Using cached response for {model}")
            return cached_response

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
                if not silent:
                    print(f"Got response from {model} after {time() - start:.2f}s")

                if use_cache:
                    cache.set(cache_key, message)
                return message
            except RateLimitError as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = (2**attempt) + random.random()
                if not silent:
                    print(f"Rate limit error: {e}. Retrying in {wait_time:.2f} seconds...")
                await asyncio.sleep(wait_time)

    raise RuntimeError("Retries exhausted")


async def get_messages_with_few_shot_prompt(
    few_shot_prompts: list[list[dict]],
    prompts: list[str],
    model: str = "sonnet-35",
    max_retries: int = 5,
) -> list[dict]:
    """Get multiple model responses in parallel with few-shot prompting."""
    if len(few_shot_prompts) != len(prompts):
        few_shot_prompts = [few_shot_prompts] * len(prompts)
    messages = await tqdm.gather(
        *[
            get_message_with_few_shot_prompt(
                few_shot_prompt, prompt=prompt, model=model, max_retries=max_retries
            )
            for few_shot_prompt, prompt in zip(few_shot_prompts, prompts)
        ],
        desc=f"Getting {model} responses",
        total=len(prompts),
    )
    return messages
