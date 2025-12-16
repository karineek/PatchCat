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
cd ../running-model
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
cd ../llm
python3 local_llm_patchDiff.py <diff-of-two-files>
```

## Full Automation from git diff to cluster
```
python3 PatchCat.py <diff-of-two-files>
```

## Automation of Model Training

Due to licensing issues, we cannot legally publish the training of the model as we received no permission from the original author of one of the libraries used to do so. 

Nonetheless, please contact us if you wish to retrain the PatchCat model. We can, at least, supply some of the script and a reference to the library you need to copy (forwhich we did not have permission to share).

We are working on a full refactoring of PatchCat to replace this library with our own code.
