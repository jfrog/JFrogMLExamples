from datasets import load_dataset
import config

def load_datasets(percentage: float):
    """
    Loads the DevOps dataset and returns train/eval splits plus a formatter.

    Args:
        percentage (float): The percentage of the dataset to use (0-100).

    Returns:
        (train_dataset, eval_dataset, formatting_func)
    """
    safe_percentage = max(min(percentage, 100.0), 0.1)
    dataset = load_dataset(config.DATASET_ID, split="train")
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    train_dataset = dataset["train"].shuffle(seed=42)
    eval_dataset = dataset["test"].shuffle(seed=42)

    if safe_percentage < 100.0:
        train_size = max(1, int(len(train_dataset) * (safe_percentage / 100)))
        eval_size = max(1, int(len(eval_dataset) * (safe_percentage / 100)))
        train_dataset = train_dataset.select(range(train_size))
        eval_dataset = eval_dataset.select(range(eval_size))

    def format_instruction(example):
        instruction = example.get("Instruction", "") or example.get("instruction", "")
        inp = example.get("Prompt", "") or example.get("prompt", "")
        response = example.get("Response", "") or example.get("response", "")
        full_prompt = f"<s>[INST] {instruction}\n{inp} [/INST] {response} </s>"
        return full_prompt

    return train_dataset, eval_dataset, format_instruction
