#!/usr/bin/env python3
import sys
import os
from functools import cache
from joblib import load
from litellm import completion
import difflib
import argparse
from pathlib import Path
import difflib

import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

## Uncomment if you want to run with Ollama API
#os.environ["OLLAMA_API_KEY"] = "9990d84828sd4a484805609942baf97c-.VxGkNKkTIT3IdZUPDBy4vgi" # Not a valid key, replace with your

# a is fromlines, and b is tolines
def make_diff(fromlines: str, tolines: str) -> str:
    return difflib.ndiff(fromlines, tolines)
    
def summarize_diff(record: str, model: str, api_base: str, max_words: int) -> str:
    """Summarise the diff text using the local LLM."""
    prompt = f"Summarize in about {max_words} words:\n\n{record}"
    try:
        response = completion(
            model=model,
            api_base=api_base,
            messages=[
                {"role": "system", "content": "Output only the summary as plain text."},
                {"role": "user", "content": prompt},
            ],
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[summary-error: {e}]"
      
def classify_text(text: str, vectorizer, clf) -> str:
    """Classify a text string using the pre-trained model."""
    clean_text = text.lower().strip()
    X = vectorizer.transform([clean_text])
    return clf.predict(X)[0]

def process_diff(diff_input: str, *, model: str, api_base: str, max_words: int, vectorizer, clf) -> str:
    """Run the complete pipeline: summarise -> classify."""
    summary = summarize_diff(diff_input, model=model, api_base=api_base, max_words=max_words)
    label = classify_text(summary, vectorizer=vectorizer, clf=clf)
    return f"[{label}] {summary}"

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="PatchCatGin.py",
        description="Given original and patched code, generate a diff and classify the summary.",
    )

    # Required input (no positional, no stdin)
    parser.add_argument(
        "--diff-text",
        help="The diff content as a string.",
    )

    # These two if diff-text was not entered
    parser.add_argument(
        "--A-text",
        help="Content of the first string.",
    )
    parser.add_argument(
        "--B-text",
        help="Content of the second string.",
    )

    if args.diff_text is not None:
        if args.A_text is not None or args.B_text is not None:
            parser.error("--diff-text cannot be used with --A-text/--B-text")
    else:
        if args.A_text is None or args.B_text is None:
            parser.error("Provide either --diff-text or both --A-text and --B-text")
        diff_text = make_diff(args.A_text, args.B_text)

    parser.add_argument("--ollama-model", default="ollama/llama3.2")
    parser.add_argument("--ollama-api-base", default="http://localhost:11434")
    # When using ollama API: either uncomment this or send these strings as paraments.
    # parser.add_argument("--ollama-model", default="ollama/gpt-oss:20b-cloud")
    # parser.add_argument("--ollama-api-base", default="https://ollama.com")
    parser.add_argument("--model-path", default=os.path.join("running-model", "model.pkl"))
    parser.add_argument("--vectorizer-path", default=os.path.join("running-model", "vectorizer.pkl"))
    parser.add_argument("--max-summary-words", type=int, default=15)

    return parser

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # Load model + vectorizer after parsing, so paths can be overridden
    try:
        vectorizer = load(args.vectorizer_path)
    except Exception as e:
        parser.error(f"Failed to load vectorizer from {args.vectorizer_path!r}: {e}")

    try:
        clf = load(args.model_path)
    except Exception as e:
        parser.error(f"Failed to load model from {args.model_path!r}: {e}")

    # Take the diff text and send to process diff check
    diff_text = args.diff_text

    if "LLM GAVE NO SUGGESTION" in diff_text or "NOT YET APPLIED" in diff_text:
        print("[1] LLM gave no suggestion")
        return 0
    
    result = process_diff(
        diff_text,
        model=args.ollama_model,
        api_base=args.ollama_api_base,
        max_words=args.max_summary_words,
        vectorizer=vectorizer,
        clf=clf,
    )
    print(result)
    return 0

# Main
if __name__ == "__main__":
    raise SystemExit(main())
