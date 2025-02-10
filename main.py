# %%
import asyncio

import nest_asyncio
from prettytable import PrettyTable

from evaluation import evaluate_model, load_questions, MATHQuestion
from prompts import get_few_shot_prompt
import numpy as np

nest_asyncio.apply()  # Allow nested event loops in Jupyter

all_train = load_questions("train")


def generate_few_shot_examples(all_examples: list[MATHQuestion], num_examples: int) -> list[dict]:
    """Generate few-shot examples from training data."""
    idxs = np.random.choice(len(all_examples), size=num_examples, replace=False)
    train_subset = [all_examples[i] for i in idxs]
    few_shot_examples = [(q.problem, q.solution) for q in train_subset]
    return get_few_shot_prompt(few_shot_examples)


# %%
num_test_examples = 50
test_data_small = load_questions("test", num_examples=num_test_examples)
# models = ["haiku-3", "sonnet-3", "sonnet-35", "opus-3"]
models = ["haiku-3", "sonnet-35"]
shots = [1, 4, 16]

table = PrettyTable()
table.field_names = ["Model", "0-shot"] + [f"{i}-shot" for i in shots]
for model in models:
    row = [model]
    print(f"Evaluating {model}...")
    zero_shot_results = asyncio.run(evaluate_model(test_data_small, model))
    row.append(f"{zero_shot_results[0]:.1%}")
    print(f"Zero-shot results: {zero_shot_results[0]:.1%}")
    for shot in shots:
        few_shot_prompts = [
            generate_few_shot_examples(all_train, shot) for _ in range(len(test_data_small))
        ]
        few_shot_results = asyncio.run(evaluate_model(test_data_small, model, few_shot_prompts))
        row.append(f"{few_shot_results[0]:.1%}")
        print(f"{shot}-shot results: {few_shot_results[0]:.1%}")
    table.add_row(row)

print(f"\nInitial Prompting Results ({num_test_examples} examples):")
print(table)

# %%
