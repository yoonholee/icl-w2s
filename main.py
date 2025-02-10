# %%
import asyncio

import nest_asyncio
import numpy as np

from evaluation import eval_model_answers, evaluate_model, load_questions
from prompts import get_few_shot_prompt

nest_asyncio.apply()  # Allows nested event loops in Jupyter

few_shot_prompt = get_few_shot_prompt(
    [("What is 2 + 2?", "2 + 2 = 4."), ("What is 49*7?", "49 * 7 = 343.")]
)
print("Few Shot Prompt Messages:")
for message in few_shot_prompt:
    role = message["role"].capitalize()
    text = message["content"][0]["text"]
    print(f"{role}: {text}")

# %%
train_dataset = load_questions("train")
test_dataset = load_questions("test")
oracle_acc = np.mean(eval_model_answers(train_dataset, [q.answer for q in train_dataset])) * 100
print(f"Oracle accuracy: {oracle_acc:.2f} (should be 100%)")

# %%
train_dataset = load_questions("train")
test_dataset = load_questions("test")

num_test_examples = 5
test_data_small = test_dataset[:: len(test_dataset) // num_test_examples]
models = ["haiku-3", "sonnet-3", "sonnet-35", "opus-3"]

for model in models:
    print(f"Evaluating {model}...")
    results = asyncio.run(evaluate_model(test_data_small, model))
    print(f"{model} accuracy: {results[0]:.1%}")
