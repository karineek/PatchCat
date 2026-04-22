# PatchCat

PatchCat, given a difference between two versions of a Java file (a code edit), returns the category of the edit to be used in the GI loop, e.g. in gintool.

## Requirements

```
sudo apt update
sudo apt install python3-venv python3-pip
sudo apt install  python3-distutils # OR pip3 install setuptools for which you will need venv too
pip3 install -r requirements.txt
python3 -m nltk.downloader punkt
python3 -m nltk.downloader punkt_tab
```
Then you will need to install Ollama and the model to be tested.

```
curl -fsSL https://ollama.com/install.sh | sh
```
Some possible models:
```
ollama pull deepseek-r1 
ollama pull gemma3 
ollama pull gemma3:27b 
ollama pull gemma3:12b 
ollama pull deepseek-coder-v2 
ollama pull gemma3:4b 
ollama pull llama3.2
```

## Use Ready Model
To run the trained model reported in ASE NIER 2025, use this:
```
cd src/running-model
python3 unseen-retrives-batch.py <Text-Short-Description-of-Patch>
```

For example:
```
python3 unseen-retrives-batch.py "adds Object variable, checks type and returns Map or throws exception."
python3 unseen-retrives-batch.py "HashMap constructor and Value/Function types changed, JsValue.fromJavaMap used instead."
python3 unseen-retrives-batch.py "Tokenizing a line and populating a command with given arguments."
```

## Create a Summary via Local LLMs
To run a summary of a diff between two Java source files, with local LLMS, use this:
```
cd src/llm
python3 local_llm_patchDiff.py <diff-of-two-files>
```

## Full Automation from git diff to cluster
```
cd src
python3 PatchCat.py <diff-of-two-files>
```

and to fully control each parameter of the deployment:
```
ubuntu@fuzzing-05:~/gin/papar-2026/PatchCat/src$ python3 PatchCatGin.py \
>   --diff-text "diff --git a/foo.py b/foo.py
> --- a/foo.py
> +++ b/foo.py
> @@ -1 +1 @@
> -print('hi')
> +print('hello')" \
>   --ollama-model "ollama/llama3.2" \
>   --ollama-api-base "http://localhost:11434" \
>   --model-path "running-model/model.pkl" \
>   --vectorizer-path "running-model/vectorizer.pkl" \
>   --max-summary-words 15
[6] Modified foo.py to print 'hello' instead of 'hi'.
ubuntu@fuzzing-05:~/gin/papar-2026/PatchCat/src$
```

## Automation of Model Training

Due to licensing issues, we cannot legally publish the training of the model as we received no permission from the original author of one of the libraries used to do so. 

Nonetheless, please contact us if you wish to retrain the PatchCat model. We can, at least, supply some of the script and a reference to the library you need to copy (forwhich we did not have permission to share).

We are working on a full refactoring of PatchCat to replace this library with our own code. Below are instructions for the already-immigrated parts.

### PatchCat – Training and Prediction Statistics Script

This README documents **only the functionality that is currently implemented and usable** in `training_PatchCat.py`.

The script supports **three operational modes**:

1. `cold` – Cold-start clustering using SentenceTransformer embeddings + anchor-initialised K-Means
2. `unseen` – Prediction on unseen data using a pre-trained *vectorizer + classifier*
3. `unseencold` – Prediction on unseen data using a trained *K-Means* model

All other options (`hot`, `mapping`) are placeholders and should be ignored for now as we are still immigrating the code to this repository.

---

### Overview

The script clusters or classifies short text items (one per line), typically representing code-change descriptions or patches.

### Command-Line Usage

#### 1. Cold Start Clustering (`cold`)

**Purpose**
- Embed all input texts using SentenceTransformer
- Cluster them into 18 anchor-guided clusters using K-Means
- Save:
  - cluster assignments
  - embeddings
  - trained K-Means model
- Evaluate against ground-truth labels (if provided)

**Command**
```bash
python3 training_PatchCat.py cold --input gin_untagged --truelabels gin_tagged --output clustered_output.tsv \
                                  --embeddings embeddings.npy --model all-MiniLM-L12-v2 --outmodel kmeans.pkl
```

**Outputs**
- `clustered_output.tsv` :
  ```
  <cluster_id>\t<text>
  ```
- `embeddings.npy`: NumPy array of sentence embeddings
- `kmeans.pkl`    : Serialised sklearn K-Means model

The script also prints a short preview (first (head) items) of each cluster to stdout.

##### Evaluation Metrics

If `--truelabels` is provided, the script reports:

1. **Clustering Accuracy**
   - Uses Hungarian matching over the confusion matrix
   - Accounts for label permutation

2. **Normalized Mutual Information (NMI)**
   - Measures agreement between true labels and clusters

Printed as:
```
[Eval] >>> Accuracy: X.XXXX
[Eval] >>> NMI:      X.XXXX
```
---

#### 2. Unseen Prediction – Full Model (`unseen`)

**Purpose**
- Apply a previously trained classical ML model
- Uses:
  - `vectorizer.pkl`
  - `model.pkl`

**Command**
```bash
python3 training_PatchCat.py unseen --input gindata/unseen-v2 --vec vectorizer.pkl --model model.pkl
```

**Output** (stdout only)
```
[cluster_id] text
```

Example:
```
[12] renamed variable foo to bar
[3] removed dead code
```

No files are written in this mode.

---

### 3. Unseen Prediction – Cold Model (`unseencold`)

**Purpose**
- Predict clusters for unseen data using the saved K-Means model
- Recomputes embeddings using the same SentenceTransformer

**Command**
```bash
python3 training_PatchCat.py unseencold --input data/unseen --model all-MiniLM-L12-v2 --coldmodel kmeans.pkl
```

**Output** (stdout only)
```
[cluster_id] text
```

No files are written in this mode.

---


## Publications

*Even-Mendoza, K., Brownlee, A., Geiger, A., Hanna, C., Petke, J., Sarro, F., & Sobania, D. (2025). LLM-Guided Genetic Improvement: Envisioning Semantic Aware Automated Software Evolution. In New Ideas and Emerging Results Track, 40th IEEE/ACM International Conference on Automated Software Engineering, ASE 2025: ASE 2025 NIER*

The arXiv version of the paper is available [here](https://arxiv.org/abs/2508.18089).

BibTex Entry:
```
@inbook{PatchCat:ASE:NIER:2025,
  title = "LLM-Guided Genetic Improvement: Envisioning Semantic Aware Automated Software Evolution",
  abstract = "Genetic Improvement (GI) of software automatically creates alternative software versions which are improved according to certain properties of interests (e.g., running-time). Search-based GI excels at navigating large program spaces, but operates primarily at syntactic level. In contrast, Large Language Models (LLMs) offer semantic-aware edits, yet lack goal-directed feedback and control (which is instead a strength of GI). As such, we propose the investigation of a new research line on AI-powered GI aimed at incorporating semantic aware search. We take a first step at it by augmenting GI with the use of automated clustering of LLM edits. We provide initial empirical evidence that our proposal, dubbed PatchCat, allows us to automatically and effectively categorize LLM-suggested patches. PatchCat identified 18 different types of software patches and categorized newly suggested patches with high accuracy. It also enabled detecting NoOp edits in advance and, prospectively, to skip test suite execution to save resources in many cases. These results, coupled with the fact that PatchCat works with small, local LLMs, are a promising step toward interpretable, efficient, and green GI. We outline a rich agenda of future work and call for the community to join our vision of building a principled understanding of LLM-driven mutations, guiding the GI search process with semantic signals.",
  author = "Karine Even-Mendoza and Alexander Brownlee and Alina Geiger and Carol Hanna and Justyna Petke and Federica Sarro and Dominik Sobania",
  year = "2025",
  month = nov,
  day = "16",
  language = "English",
  booktitle = "New Ideas and Emerging Results Track, 40th IEEE/ACM International Conference on Automated Software Engineering, ASE 2025",
}
```

*Even Mendoza, K., Brownlee, A., Geiger, A., Hanna, C., Petke, J., Sarro, F., & Sobania, D. (2025). Artifact of LLM-Guided Genetic Improvement: Envisioning Semantic Aware Automated Software Evolution (ASE 2025 V1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.15834984*

BibTex Entry:
```
@dataset{even_mendoza_2025_15834984,
  author       = {Even Mendoza, Karine and
                  Brownlee, Alexander and
                  Geiger, Alina and
                  Hanna, Carol and
                  Petke, Justyna and
                  Sarro, Federica and
                  Sobania, Dominik},
  title        = {Artifact of LLM-Guided Genetic Improvement:
                   Envisioning Semantic Aware Automated Software
                   Evolution
                  },
  month        = jul,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {ASE 2025 V1},
  doi          = {10.5281/zenodo.15834984},
  url          = {https://doi.org/10.5281/zenodo.15834984},
}
```
