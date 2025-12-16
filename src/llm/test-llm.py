from functools import cache
import json
import argparse
from litellm import completion
import re
from typing import Iterable
from codecarbon import EmissionsTracker
import datetime
import os 

def task_with_litellm(record: str, model: str = "ollama/llama3.2", max_words: int = 15) -> str:
    prompt = f"Summarize in about {max_words} words:\n\n{record}"
    try:
        response = completion(
            model=model,
            api_base="http://localhost:11434",
            messages=[
                {"role": "system", "content": "Output only the summary as plain text."},
                {"role": "user", "content": prompt}
            ],
        )
        content = response["choices"][0]["message"]["content"]
        return content
    except Exception as e:
        return f"[summary-error: {e}]"

# allow the functions to be used from the command line
if __name__ == "__main__":
    with open("cluster_true.tsv", "w", encoding="utf-8") as f_out:
        tsv_reader = csv.reader(file, delimiter='\t')
        for record in tsv_reader:
            res = task_with_litellm(record, model=model)
            print(f"[{record}] {res}")

