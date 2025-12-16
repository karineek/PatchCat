#!/usr/bin/env python3
import sys
import os
from functools import cache
from joblib import load
from litellm import completion
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

# ---------- Hyper-parameters ----------
OLLAMA_MODEL = "ollama/llama3.2"
OLLAMA_API_BASE = "http://localhost:11434"
MODEL_PATH = os.path.join("running-model", "model.pkl")
VECTORIZER_PATH = os.path.join("running-model", "vectorizer.pkl")
MAX_SUMMARY_WORDS = 15
# --------------------------------------

# Load model + vectorizer once
vectorizer = load(VECTORIZER_PATH)
clf = load(MODEL_PATH)

def summarize_diff(record: str, model: str = OLLAMA_MODEL, max_words: int = MAX_SUMMARY_WORDS) -> str:
    """Summarise the diff text using the local LLM."""
    prompt = f"Summarize in about {max_words} words:\n\n{record}"
    try:
        response = completion(
            model=model,
            api_base=OLLAMA_API_BASE,
            messages=[
                {"role": "system", "content": "Output only the summary as plain text."},
                {"role": "user", "content": prompt}
            ],
        )
        return response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[summary-error: {e}]"


def classify_text(text: str) -> str:
    """Classify a text string using the pre-trained model."""
    clean_text = text.lower().strip()
    X = vectorizer.transform([clean_text])
    return clf.predict(X)[0]


def process_diff(diff_input: str) -> str:
    """Run the complete pipeline: summarise -> classify."""
    summary = summarize_diff(diff_input)
    label = classify_text(summary)
    return f"[{label}] {summary}"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 process_diff_and_classify.py <diff-file or diff-text>")
        sys.exit(1)

    diff_arg = sys.argv[1]

    # If the argument is a file path, read it
    if os.path.exists(diff_arg):
        with open(diff_arg, "r", encoding="utf-8") as f:
            diff_text = f.read()
    else:
        diff_text = diff_arg

    result = process_diff(diff_text)
    print(result)
