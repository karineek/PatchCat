# PatchCat Training Commands

## Prerequisites

- Python 3
- All project dependencies installed (whatever your repo requires)
- Input data available:
  - `gin_untagged` (unlabeled inputs for clustering)
  - `gin_tagged` (true labels used for evaluation reporting)
- Script available:
  - `training_PatchCat.py` and `patchCat_clustering-alt.py`

Quick sanity checks:

```bash
ls -lah gin_untagged gin_tagged
python3 training_PatchCat.py -h
```

---

## Training (cold start)

### A) Sentence-Transformer embeddings: `all-MiniLM-L12-v2`

#### 1) KMeans
```bash
python3 training_PatchCat.py cold \
  --input gin_untagged \
  --truelabels gin_tagged \
  --model all-MiniLM-L12-v2 \
  --ML kmeans \
  --output kmeans-clustered_output.tsv \
  --outmodel kmeans.pkl \
  --embeddings embeddings-kmeans.npy
```

#### 2) KMeans++
```bash
python3 training_PatchCat.py cold \
  --input gin_untagged \
  --truelabels gin_tagged \
  --model all-MiniLM-L12-v2 \
  --ML kmeans++ \
  --output kmeans++-clustered_output.tsv \
  --outmodel kmeans++.pkl \
  --embeddings embeddings-kmeans++.npy
```

#### 3) COP-KMeans
```bash
python3 training_PatchCat.py cold \
  --input gin_untagged \
  --truelabels gin_tagged \
  --model all-MiniLM-L12-v2 \
  --ML copkmeans \
  --output copkmeans-clustered_output.tsv \
  --outmodel copkmeans.pkl \
  --embeddings embeddings-copkmeans.npy
```

#### 4) Spherical KMeans (`skmeans`)
```bash
python3 training_PatchCat.py cold \
  --input gin_untagged \
  --truelabels gin_tagged \
  --model all-MiniLM-L12-v2 \
  --ML skmeans \
  --output skmeans-clustered_output.tsv \
  --outmodel skmeans.pkl \
  --embeddings embeddings-skmeans.npy
```

---

### B) TF-IDF features: `tfidf`

#### 1) KMeans (TF-IDF)
```bash
python3 training_PatchCat.py cold \
  --input gin_untagged \
  --truelabels gin_tagged \
  --model tfidf \
  --ML kmeans \
  --output kmeans-tfidf-clustered_output.tsv \
  --outmodel kmeans-tfidf.pkl \
  --embeddings embeddings-kmeans-tfidf.npy
```

#### 2) KMeans++ (TF-IDF)
```bash
python3 training_PatchCat.py cold \
  --input gin_untagged \
  --truelabels gin_tagged \
  --model tfidf \
  --ML kmeans++ \
  --output kmeans++-tfidf-clustered_output.tsv \
  --outmodel kmeans++-tfidf.pkl \
  --embeddings embeddings-kmeans++-tfidf.npy
```

#### 3) COP-KMeans (TF-IDF)
```bash
python3 training_PatchCat.py cold \
  --input gin_untagged \
  --truelabels gin_tagged \
  --model tfidf \
  --ML copkmeans \
  --output copkmeans-tfidf-clustered_output.tsv \
  --outmodel copkmeans-tfidf.pkl \
  --embeddings embeddings-copkmeans-tfidf.npy
```

---

## Optional: Unseen evaluation (`unseencold`)

These commands are only meaningful if you have **unseen data** and corresponding **manual labels**.

### 1) Evaluate `kmeans++` model
```bash
python3 training_PatchCat.py unseencold \
  --input data/unseen-v2 \
  --model all-MiniLM-L12-v2 \
  --coldmodel kmeans++.pkl \
  --truelabels data/tagged-v2.fixed.txt
```

### 2) Evaluate `copkmeans` model
```bash
python3 training_PatchCat.py unseencold \
  --input data/unseen-v2 \
  --model all-MiniLM-L12-v2 \
  --coldmodel copkmeans.pkl \
  --truelabels data/tagged-v2.fixed.txt
```

### 3) Evaluate `skmeans` model
```bash
python3 training_PatchCat.py unseencold \
  --input data/unseen-v2 \
  --model all-MiniLM-L12-v2 \
  --coldmodel skmeans.pkl \
  --truelabels data/tagged-v2.fixed.txt
```

---


## Output:
```
%% python3 patchCat_clustering.py cold --input gin_untagged --truelabels gin_tagged \
%% --model all-MiniLM-L12-v2 --ML kmeans \
%% --output kmeans-clustered_output.tsv \
%% --outmodel kmeans.pkl --embeddings embeddings-kmeans.npy 
%% [K-means] >>> Saved model to kmeans.pkl
%% [Eval] >>> Read 5806 TRUE labels.
%% [Eval] >>> Accuracy: 0.8229
%% [Eval] >>> NMI:      0.7628

%% python3 patchCat_clustering.py cold --input gin_untagged --truelabels gin_tagged --model tfidf --ML kmeans --output kmeans-tfidf-clustered_output.tsv --outmodel kmeans-tfidf.pkl --embeddings embeddings-kmeans-tfidf.npy
%% [TF-IDF] >>> Saved clustering model to kmeans-tfidf.pkl
%% [TF-IDF] >>> Saved vectorizer bundle to vectorizer.pkl
%% [Eval] >>> Read 5806 TRUE labels.
%% [Eval] >>> Accuracy: 0.7478
%% [Eval] >>> NMI:      0.6958

%% This is not working as we do not have unseen data we manually tagged, but if we have during experiments, we can!
%% python3 patchCat_clustering.py unseencold --input data/unseen-v2 --model all-MiniLM-L12-v2 --coldmodel kmeans++.pkl --truelabels data/tagged-v2.fixed.txt
%%%=============================

%% python3 patchCat_clustering.py cold --input gin_untagged --truelabels gin_tagged \
%% --model all-MiniLM-L12-v2 --ML kmeans++ \
%% --output kmeans++-clustered_output.tsv \
%% --outmodel kmeans++.pkl --embeddings embeddings-kmeans++.npy 
%% [K-means] >>> Saved model to kmeans++.pkl
%% [Eval] >>> Read 5806 TRUE labels.
%% [Eval] >>> Accuracy: 0.7713
%% [Eval] >>> NMI:      0.7401

%% python3 patchCat_clustering.py cold --input gin_untagged --truelabels gin_tagged --model tfidf --ML kmeans++ --output kmeans++-tfidf-clustered_output.tsv --outmodel kmeans++-tfidf.pkl --embeddings embeddings-kmeans++-tfidf.npy
%% [TF-IDF] >>> Saved clustering model to kmeans++-tfidf.pkl
%% [TF-IDF] >>> Saved vectorizer bundle to vectorizer.pkl
%% [Eval] >>> Read 5806 TRUE labels.
%% [Eval] >>> Accuracy: 0.4874
%% [Eval] >>> NMI:      0.5163

%% This is not working as we do not have unseen data we manually tagged, but if we have during experiments, we can!
%% python3 patchCat_clustering.py unseencold --input data/unseen-v2 --model all-MiniLM-L12-v2 --coldmodel kmeans++.pkl --truelabels data/tagged-v2.fixed.txt

%%%=============================
%% python3 patchCat_clustering.py cold --input gin_untagged --truelabels gin_tagged \
%% --model all-MiniLM-L12-v2 --ML copkmeans \
%% --output copkmeans-clustered_output.tsv \
%% --outmodel copkmeans.pkl --embeddings embeddings-copkmeans.npy 
%% [COP-K-MEANS] >>> Saved centroid model to copkmeans.pkl
%% [Eval] >>> Read 5806 TRUE labels.
%% [Eval] >>> Accuracy: 0.8290
%% [Eval] >>> NMI:      0.7717

%% python3 patchCat_clustering.py cold --input gin_untagged --truelabels gin_tagged --model tfidf --ML copkmeans --output copkmeans-tfidf-clustered_output.tsv --outmodel copkmeans-tfidf.pkl --embeddings embeddings-copkmeans-tfidf.npy
%% [TF-IDF-COP-K-MEANS] >>> Saved centroid model to copkmeans-tfidf.pkl
%% [TF-IDF] >>> Saved vectorizer bundle to vectorizer.pkl
%% [Eval] >>> Read 5806 TRUE labels.
%% [Eval] >>> Accuracy: 0.7501
%% [Eval] >>> NMI:      0.6952

%% This is not working as we do not have unseen data we manually tagged, but if we have during experiments, we can!
%% python3 patchCat_clustering.py unseencold --input data/unseen-v2 --model all-MiniLM-L12-v2 --coldmodel copkmeans.pkl --truelabels data/tagged-v2.fixed.txt


%%%=============================
%% python3 patchCat_clustering.py cold --input gin_untagged --truelabels gin_tagged \
%% --model all-MiniLM-L12-v2 --ML skmeans \
%% --output skmeans-clustered_output.tsv \
%% --outmodel skmeans.pkl --embeddings embeddings-skmeans.npy 
%% [K-means] >>> Saved model to skmeans.pkl
%% [Eval] >>> Read 5806 TRUE labels.
%% [Eval] >>> Accuracy: 0.8229
%% [Eval] >>> NMI:      0.7628

%% This is not working as we do not have unseen data we manually tagged, but if we have during experiments, we can!
%% python3 patchCat_clustering.py unseencold --input data/unseen-v2 --model all-MiniLM-L12-v2 --coldmodel skmeans.pkl --truelabels data/tagged-v2.fixed.txt
```
