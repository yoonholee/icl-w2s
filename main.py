# %%
import asyncio
import copy

import nest_asyncio
import numpy as np
from prettytable import PrettyTable

from evaluation import MATHQuestion, evaluate_model, load_questions
from prompts import get_few_shot_prompt

nest_asyncio.apply()  # Allow nested event loops in Jupyter

num_test_examples = None
test_data_small = load_questions("test", num_examples=num_test_examples)

all_train = load_questions("train")
split_num = int(len(all_train) * 0.5)
train_idxs, val_idxs = np.split(np.random.permutation(len(all_train)), [split_num])
train_data = [all_train[i] for i in train_idxs]
val_data = [all_train[i] for i in val_idxs]
num_val_examples = 200
val_data_small = val_data[:num_val_examples]

print(f"Train: {len(train_data)}, Val: {len(val_data_small)}, Test: {len(test_data_small)}")


def generate_few_shot_examples(
    all_examples: list[MATHQuestion], num_examples: int, w2s_prompt: bool = False
) -> list[dict]:
    """Generate few-shot examples from training data."""
    idxs = np.random.choice(len(all_examples), size=num_examples, replace=False)
    train_subset = [all_examples[i] for i in idxs]
    few_shot_examples = [(q.problem, q.solution) for q in train_subset]
    return get_few_shot_prompt(few_shot_examples, w2s_prompt)


num_shots = 4
few_shot_prompts = [
    generate_few_shot_examples(train_data, num_shots) for _ in range(num_val_examples)
]

print(
    f"\nEvaluating haiku-3 on {num_val_examples} validation examples with {num_shots}-shot prompting..."
)
val_acc, val_responses = asyncio.run(evaluate_model(val_data_small, "haiku-3", few_shot_prompts))
print(f"Validation accuracy: {val_acc:.1%}")

val_dataset_w2s = copy.deepcopy(val_data_small)
for val_response, val_question in zip(val_responses, val_dataset_w2s):
    val_question.solution = val_response

few_shot_prompts_w2s = [
    generate_few_shot_examples(val_dataset_w2s, num_shots) for _ in range(len(val_dataset_w2s))
]
w2s_results, _ = asyncio.run(evaluate_model(test_data_small, "sonnet-35", few_shot_prompts_w2s))
print(f"Test accuracy with {num_shots}-shot: {w2s_results:.1%}")

# %%
# models = ["haiku-3", "sonnet-3", "sonnet-35", "opus-3"]
models = ["haiku-3", "sonnet-35"]
shots = [1, 4]

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
            generate_few_shot_examples(val_data_small, shot) for _ in range(len(test_data_small))
        ]
        few_shot_results = asyncio.run(evaluate_model(test_data_small, model, few_shot_prompts))
        row.append(f"{few_shot_results[0]:.1%}")
        print(f"{shot}-shot results: {few_shot_results[0]:.1%}")
    table.add_row(row)

print(f"\nInitial Prompting Results ({num_test_examples} examples):")
print(table)

# %%
