#!/usr/bin/env python3
import sys
import os
from functools import cache
from joblib import load
from litellm import completion
import difflib
import argparse
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

## Uncomment if you want to run with Ollama API
#os.environ["OLLAMA_API_KEY"] = "9990d84828sd4a484805609942baf97c-.VxGkNKkTIT3IdZUPDBy4vgi" # Not a valid key, replace with your

# a is from lines, and b is to lines
import subprocess
import tempfile
import os

def make_diff(a: str, b: str) -> str:
    f1 = tempfile.NamedTemporaryFile("w", delete=False)
    f2 = tempfile.NamedTemporaryFile("w", delete=False)
    try:
        f1.write(a)
        f2.write(b)
        f1.close()
        f2.close()

        result = subprocess.run(
            ["diff", f1.name, f2.name],
            capture_output=True,
            text=True
        )
        return result.stdout

    finally:
        os.unlink(f1.name)
        os.unlink(f2.name)

def llm_classifcation(diff_input: str, *, model: str, api_base: str, max_words: int, only_sum=0) -> str:
    """Run the complete pipeline: summarise -> classify."""
    summary = summarize_diff(diff_input, model=model, api_base=api_base, max_words=max_words)

    if (not only_sum): # We send both the summary and the code diff of the patch
        res = direct_diff_classification("This pathc is doing " + summary + " and I extract also the actual code diff of the patch" + diff_input,
                                           model, api_base, -1)
    else: # only the summary of the diff is sent to the LLM
        res = direct_diff_classification(summary,
                                           model, api_base, 0)
    return f"{res} {summary}"


def direct_diff_classification(record: str, model: str, api_base: str, direct=1) -> str:
    """
        Classify a patch (given as a diff) directly.
        Returns:
            0 = safe/meaningful patch, no need to test
            1 = sensible patch, but compile + test suite recommended
            2 = rubbish/trivial patch, e.g. comments-only, whitespace-only, dead code, noise
        If the LLM response cannot be parsed, it defaults to 1.
    """

    ## The prompt for this specific action:
    if (not direct): # When we use the summary, not the code
        prompt = f"Classify the following summary of code diff into exactly one category 0, 1, or 2:\n\n{record}"
    elif direct < 0: # Negative is patch diff + summary
        prompt = f"Classify the following code diff with its explanation into exactly one category 0, 1, or 2:\n\n{record}"
    else: # Positive is patch diff only (direct use of LLMs)
        prompt = f"Classify the following code diff into exactly one category 0, 1, or 2:\n\n{record}"
    text = "No Response"
    
    ## Exec the prompt:
    try:
        response = completion(
            model=model,
            api_base=api_base,
            request_timeout=45,
            messages=[
                {
                    "role": "system",
                    "content": (f"""Classify the following code diff into exactly one category:
                        0 = The patch is clearly okay and meaningful. It can be accepted directly without extra checking.
                        1 = The patch is sensible, but may contain mistakes. It should be compiled and tested.
                        2 = The patch is rubbish/trivial/noise, such as comments-only changes, whitespace-only changes,
                            formatting-only changes, dead code, or changes that do not meaningfully affect behavior.
                        
                        Important:
                        - Output ONLY one integer: 0, 1, or 2.
                        - Do not explain.
                        - Do not include punctuation.
                        """
                        "You are a Git diff classifier. "
                        "You must classify diffs into exactly one label: 0, 1, or 2. "
                        "Output only the integer label."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        text = response["choices"][0]["message"]["content"].strip()
    
        ## Return the results:
        if text == "0":
            return "A"
        elif text == "2":
            return "C"
        elif text == "1":
            return "B"
        else:
            return "B"
            
    except Exception:
        return "B" # In case of an error, we are back to square 1, then return B to test the patch.
    
def summarize_diff(record: str, model: str, api_base: str, max_words: int) -> str:
    """Summarise the diff text using the local LLM."""
    prompt = f"Summarize in about {max_words} words:\n\n{record}"
    try:
        response = completion(
            model=model,
            api_base=api_base,
            request_timeout=45,  # seconds
            messages=[
                {"role": "system", "content": "You are Git Diff Analyser. You get output of diff of two files '<' (old) and '>' (new) with '---' separating between the codes. Output only the summary as plain text. You got 45 seconds for this task."},
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

    parser.add_argument("--ollama-model", default="ollama/llama3.2")
    parser.add_argument("--ollama-api-base", default="http://localhost:11434")
    # When using ollama API: either uncomment this or send these strings as paraments.
    # parser.add_argument("--ollama-model", default="ollama/gpt-oss:20b-cloud")
    # parser.add_argument("--ollama-api-base", default="https://ollama.com")
    parser.add_argument("--model-path", default=os.path.join("running-model", "model.pkl"))
    parser.add_argument("--vectorizer-path", default=os.path.join("running-model", "vectorizer.pkl"))
    parser.add_argument("--max-summary-words", type=int, default=15)
    parser.add_argument("--options", type=int, default=0) # change it here or from gin

    return parser

def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.diff_text is not None:
        if args.A_text is not None or args.B_text is not None:
            parser.error("--diff-text cannot be used with --A-text/--B-text")
        # Take the diff text and send to process diff check
        diff_text = args.diff_text
    else:
        if args.A_text is None or args.B_text is None:
            parser.error("Provide either --diff-text or both --A-text and --B-text")
        diff_text = make_diff(args.A_text, args.B_text)
        if "LLM GAVE NO SUGGESTION" in args.A_text or "NOT YET APPLIED" in args.A_text or "LLM GAVE NO SUGGESTION" in args.B_text or "NOT YET APPLIED" in args.B_text:
            print("[1] LLM gave no suggestion", flush=True)
            print(diff_text, flush=True)
            return 0

    # Load model + vectorizer after parsing, so paths can be overridden
    if args.options == 0: # Direct request from LLMs
        result = direct_diff_classification(
            diff_text,
            model=args.ollama_model,
            api_base=args.ollama_api_base)
        print(result, flush=True) # IF we are working with a multi-processor env./server/against server, this is needed for sync.
        return 0

    elif args.options == 1: # Direct LLM, but LLM gets the summary of the patch
        result = llm_classifcation(
            diff_input=diff_text,
            model=args.ollama_model,
            api_base=args.ollama_api_base,
            max_words=args.max_summary_words,
            only_sum=1)

        print(result, flush=True) # IF we are working with a multi-processor env./server/against server, this is needed for sync.
        return 0

    elif args.options == 2: # Direct LLM, but LLM gets the summary of the patch with the patch diff too
        result = llm_classifcation(
            diff_input=diff_text,
            model=args.ollama_model,
            api_base=args.ollama_api_base,
            max_words=args.max_summary_words, 
            only_sum=1)

        print(result, flush=True) # IF we are working with a multi-processor env./server/against server, this is needed for sync.
        return 0

    else: # Original code
        try:
            vectorizer = load(args.vectorizer_path)
        except Exception as e:
            parser.error(f"Failed to load vectorizer from {args.vectorizer_path!r}: {e}")
    
        try:
            clf = load(args.model_path)
        except Exception as e:
            parser.error(f"Failed to load model from {args.model_path!r}: {e}")
        
        result = process_diff(
            diff_text,
            model=args.ollama_model,
            api_base=args.ollama_api_base,
            max_words=args.max_summary_words,
            vectorizer=vectorizer,
            clf=clf,
        )
        print(result, flush=True) # IF we are working with a multi-processor env./server/against server, this is needed for sync.
        print(diff_text, flush=True) # same
        return 0

# Main
if __name__ == "__main__":
    raise SystemExit(main())
