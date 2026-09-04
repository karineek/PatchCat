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

# for kmeans variants
import numpy as np
from sentence_transformers import SentenceTransformer

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


def direct_diff_classification(record: str, model: str, api_base: str, direct=1, adv=0) -> str:
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

    if (adv):
        prompt = prompt + ". Please, when it is sensible, select 0. Do not cheat. Avoid hallucinations."
    
    ## Exec the prompt:
    # print(f"=== PatchCat running with: {model} via {api_base}")
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
                            formatting-only changes, dead code, or changes that do not meaningfully affect behaviour.
                        
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
            #label = "17" # If using mid version of gin-llm
            label =  "A"
        elif text == "2":
            #label = "2" # If using mid version of gin-llm
            label =  "C"
        elif text == "1":
            #label = "0" # If using mid version of gin-llm
            label =  "B"
        else:
            #label = "0" # If using mid version of gin-llm
            label =  "B"
            
    except Exception:
        #label = "0" # If using mid version of gin-llm
        label = "B" # In case of an error, we are back to square 1, then return B to test the patch.

    return f"[{label}] {text}"
    
def summarize_diff(record: str, model: str, api_base: str, max_words: int) -> str:
    """Summarise the diff text using the local LLM."""
    prompt = f"Summarize in about {max_words} words:\n\n{record}"
    try:
        response = completion(
            model=model,
            api_base=api_base,
            request_timeout=45,  # seconds
            messages=[
                {"role": "system", "content": "You are Git Diff Analyser. You get output of diff of two files '<' (old) and '>' (new) with '---' separating between the codes. Output only the summary as plain text. Never discuss comments unless they are the only change in the diff. You got 45 seconds for this task."},
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

# K-means and Co-k-means
from training_PatchCat import cop_predict_using_original_code 
def process_diff_kmeans(diff_input: str, *, model: str, api_base: str, max_words: int, 
                        model_name="all-MiniLM-L12-v2", 
                        model_path="kmeans.pkl",
                        is_cop=1) -> str:
    '''
        We support two of the models: KMEANS and CO-KMEANS
        1)  %% --model all-MiniLM-L12-v2 --ML kmeans \
            %% --output kmeans-clustered_output.tsv \
            %% --outmodel kmeans.pkl --embeddings embeddings-kmeans.npy 

            AND

        2)  %% --model all-MiniLM-L12-v2 --ML copkmeans \
            %% --output copkmeans-clustered_output.tsv \
            %% --outmodel copkmeans.pkl --embeddings embeddings-copkmeans.npy 
    '''
    summary = summarize_diff(diff_input, model=model, api_base=api_base, max_words=max_words)
                            
    model = SentenceTransformer(model_name)
    embedding = model.encode(
        [summary.lower().strip()],
        show_progress_bar=False,
    )

    clustering_model = joblib.load(model_path)
    if not is_cop:
        label = clustering_model.predict(embedding)
    else:
        # COP-KMeans model is stored as a dictionary of centroids.
        centers = np.asarray(clustering_model["centers"], dtype=float)
        eps = float(clustering_model.get("eps", 1e-12))
        
        X = np.asarray(embedding, dtype=float)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)
        label = cop_predict_using_original_code(centers, X)

    label = (label + 1) % n_clusters 
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
    parser.add_argument("--vectorizer-path", default=os.path.join("running-model", "vectorizer.pkl")) # or all-MiniLM-L12-v2 
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

        if len(args.A_text) == 0:
            print(f"[1] LLM gave no suggestion (PatchCAT) A is empty and B:{args.B_text}", flush=True)
            print(diff_text, flush=True)
            return 0

        if len(args.B_text) == 0:
            print(f"[1] LLM gave no suggestion (PatchCAT) A:{args.A_text} and B is empty", flush=True)
            print(diff_text, flush=True)
            return 0

        if "LLM GAVE NO SUGGESTION" in args.A_text or "NOT YET APPLIED" in args.A_text or "LLM GAVE NO SUGGESTION" in args.B_text or "NOT YET APPLIED" in args.B_text:
            print(f"[1] LLM gave no suggestion (PatchCAT) A:{args.A_text} and B:{args.B_text}", flush=True)
            print(diff_text, flush=True)
            return 0

        diff_text = make_diff(args.A_text, args.B_text)


    if len(diff_text) == 0:
        print(f"[1] LLM gave no suggestion (PatchCAT): Diff Text is empty", flush=True)
        print(diff_text, flush=True)
        return 0


    # Load model + vectorizer after parsing, so paths can be overridden
    # Option 0 and anything else not below is the original PatchCat functionality
    if args.options == 1: # Direct request from LLMs
        result = direct_diff_classification(
            diff_text,
            model=args.ollama_model,
            api_base=args.ollama_api_base)
        print(result, flush=True) # IF we are working with a multi-processor env./server/against server, this is needed for sync.
        return 0

    elif args.options == 2: # Direct LLM, but LLM gets the summary of the patch
        result = llm_classifcation(
            diff_input=diff_text,
            model=args.ollama_model,
            api_base=args.ollama_api_base,
            max_words=args.max_summary_words,
            only_sum=1)

        print(result, flush=True) # IF we are working with a multi-processor env./server/against server, this is needed for sync.
        return 0

    elif args.options == 3: # Direct LLM, but LLM gets the summary of the patch with the patch diff too
        result = llm_classifcation(
            diff_input=diff_text,
            model=args.ollama_model,
            api_base=args.ollama_api_base,
            max_words=args.max_summary_words, 
            only_sum=0)

        print(result, flush=True) # IF we are working with a multi-processor env./server/against server, this is needed for sync.
        return 0

    elif args.options == 4: # KMEANs variants
        result = process_diff_kmeans(
            diff_input=diff_text,
            model=args.ollama_model,
            api_base=args.ollama_api_base,
            max_words=args.max_summary_words, 
            model_name=args.vectorizer_path,
            model_path=args.model_path,
            is_cop=0)

        print(result, flush=True) # IF we are working with a multi-processor env./server/against server, this is needed for sync.
        return 0

    elif args.options == 5: # CO-KMEANS variants
        result = process_diff_kmeans(
            diff_input=diff_text,
            model=args.ollama_model,
            api_base=args.ollama_api_base,
            max_words=args.max_summary_words, 
            model_name=args.vectorizer_path,
            model_path=args.model_path,
            is_cop=1)

        print(result, flush=True) # IF we are working with a multi-processor env./server/against server, this is needed for sync.
        return 0
    
    else: # Original code, 0 or >5
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
