# PatchCat



## Requirements

```
sudo apt update
sudo apt install python3.10-venv python3.10-distutils python3-pip
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

## Automation of Model Training

Due to licensing issues, we cannot legally publish the training of the model as we received no permission from the original author of one of the libraries used to do so. 

Nonetheless, please contact us if you wish to retrain the PatchCat model. We can, at least, supply some of the script and a reference to the library you need to copy (forwhich we did not have permission to share).

We are working on a full refactoring of PatchCat to replace this library with our own code.

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
