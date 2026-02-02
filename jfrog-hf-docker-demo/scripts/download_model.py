"""
Download a Hugging Face model and save it to a local path.
Run at Docker build time so the model is baked into the image.
Default: google/flan-t5-small (text-to-text).
"""
import argparse
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        default="google/flan-t5-small",
        help="Hugging Face model ID (e.g. google/flan-t5-small)",
    )
    parser.add_argument(
        "--output-dir",
        default="/app/model",
        help="Directory to save the model",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Downloading {args.model_id} to {args.output_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id)
    tokenizer.save_pretrained(args.output_dir)
    model.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
