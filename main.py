# %%
import asyncio
import copy

import nest_asyncio
import numpy as np

from evaluation import MATHQuestion, evaluate_model, load_questions
from prompts import get_few_shot_prompt

nest_asyncio.apply()  # Allow nested event loops in Jupyter

num_test_examples = None
# num_test_examples = 100
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
    if num_examples is None:
        return []
    """Generate few-shot examples from training data."""
    idxs = np.random.choice(len(all_examples), size=num_examples, replace=False)
    train_subset = [all_examples[i] for i in idxs]
    few_shot_examples = [(q.problem, q.solution) for q in train_subset]
    return get_few_shot_prompt(few_shot_examples, w2s_prompt)


def w2s_experiment(weak_model: str, strong_model: str, num_shots: int):
    gold_fewshot_train_for_val = [
        generate_few_shot_examples(train_data, num_shots) for _ in range(num_val_examples)
    ]
    print(
        f"\nEvaluating {weak_model} on {num_val_examples} validation examples with {num_shots}-shot prompting..."
    )
    val_acc, val_responses = asyncio.run(
        evaluate_model(val_data_small, weak_model, gold_fewshot_train_for_val)
    )
    print(f"{weak_model} val acc: {val_acc:.1%}")

    val_dataset_w2s = copy.deepcopy(val_data_small)
    for val_response, val_question in zip(val_responses, val_dataset_w2s):
        val_question.solution = val_response
    weak_fewshot_val_for_test = [
        generate_few_shot_examples(val_dataset_w2s, num_shots, w2s_prompt=True)
        for _ in range(len(test_data_small))
    ]
    w2s_results, _ = asyncio.run(
        evaluate_model(test_data_small, strong_model, weak_fewshot_val_for_test)
    )
    print(f"{strong_model} with {weak_model} {num_shots}-shot: {w2s_results:.1%}")

    gold_fewshot_val_for_test = [
        generate_few_shot_examples(val_data_small, num_shots) for _ in range(len(test_data_small))
    ]
    test_acc, test_responses = asyncio.run(
        evaluate_model(test_data_small, weak_model, gold_fewshot_val_for_test)
    )
    print(f"{weak_model} gold {num_shots}-shot test acc: {test_acc:.1%}")

    gold_fewshot_val_for_test = [
        generate_few_shot_examples(val_data_small, num_shots) for _ in range(len(test_data_small))
    ]
    w2s_results, _ = asyncio.run(
        evaluate_model(test_data_small, strong_model, gold_fewshot_val_for_test)
    )
    print(f"{strong_model} with gold {num_shots}-shot: {w2s_results:.1%}")


def zero_shot_experiment(model: str):
    zero_shot_results, _ = asyncio.run(evaluate_model(test_data_small, model))
    print(f"{model} zero-shot: {zero_shot_results:.1%}")


weak_model = "haiku-3"
# strong_model = "sonnet-3"
strong_model = "sonnet-35"
w2s_experiment(weak_model=weak_model, strong_model=strong_model, num_shots=1)
zero_shot_experiment(model=weak_model)
zero_shot_experiment(model=strong_model)
# zero_shot_experiment(model="sonnet-3")
