# COLD: (K-MEANS only) python3 patchCat_clustering.py cold --input gin_untagged --output test_012026_clustered_output.tsv --model all-MiniLM-L12-v2 --truelabels  gin_tagged
# Cold unseen prediction: (K-MEANS only) python3 patchCat_clustering.py unseencold --input data/unseen --model all-MiniLM-L12-v2 --coldmodel kmeans.pkl  
# Unseen prediction: (Full model)  python3 patchCat_clustering.py unseen --input data/unseen-v2 --vec vectorizer.pkl --model model.pkl

# This Python program contains two separate ML algorithms: 
# KMeans (as in Python libraries) and semantic clustering from the paper
# TODO: add the paper.
import argparse
import numpy as np
import joblib
import os

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

# Define anchor sentences for each cluster (must be exactly 18)
anchor_sentences = [
  "Identical files: No differences observed", #1
  "comment comments", #2
  "Deleted", #3
  "repeat code repetition dup duplicated", #4
  "return statements", #5
  "caller callee", #6
  "data types, type usage and generics", #7
  "inline comparator lambda", #8
  "exception-handling constructs", #9
  "brackets", #10
  "redundant synchronised", #11
  "variable name", #12
  "control flow structure (if, for, while, ternary)", #13
  "object (including primitive types)", #14
  "Split a statement", #15
  "Swapped a + b to b + a leading to different order of evaluation", #16
  "deadcode", # 17
  "code from GitHub" #18 ==> turned to be #0 in the final version
]
assert len(anchor_sentences) == 18, "You must provide exactly 18 anchor sentences."

# Start the main - each part is separated to allow simulation for evaluation.
def main():
    print(">> Start Clustering...")

    # Are we now doing K-means (step 1) or semantic clustering (step 2)?
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="algo", required=True)

    # ---- cold only ----
    cold = subparsers.add_parser("cold", help="Cold start clustering")
    cold.add_argument("--input", default="gin_untagged")
    cold.add_argument("--truelabels", default="gin_tagged")
    cold.add_argument("--output", default="clustered_output.tsv")
    cold.add_argument("--embeddings", default="embeddings.npy")
    cold.add_argument("--model", default="all-MiniLM-L12-v2")
    cold.add_argument("--outmodel", default="kmeans.pkl")

    # ---- hot / mapping (placeholders for now) ----
    subparsers.add_parser("hot", help="Hot clustering")
    subparsers.add_parser("mapping", help="Mapping phase")

    # ---- unseen only ----
    unseen = subparsers.add_parser("unseen", help="Prediction of unseen data")
    unseen.add_argument("--input", default="gindata/unseen-v2")
    unseen.add_argument("--vec", default="vectorizer.pkl")
    unseen.add_argument("--model", default="model.pkl")

    # ---- unseen cold only ----
    unseen_cold = subparsers.add_parser("unseencold", help="Prediction of unseen data with K-Means model")
    unseen_cold.add_argument("--input", default="gindata/unseen-v2")
    unseen_cold.add_argument("--model", default="all-MiniLM-L12-v2")
    unseen_cold.add_argument("--coldmodel", default="kmeans.pkl")

    # Parse
    args = parser.parse_args()


    # cold start phase
    if args.algo == "cold":
        run_kmeans_pipeline(
            input_file=args.input,
            true_labels=args.truelabels,
            output_file=args.output,
            output_model=args.outmodel,
            embeddings_file=args.embeddings,
            model_name=args.model,
            anchors=anchor_sentences,
            n_clusters=18
        )

    # hot start phase
    elif args.algo == "hot":
        run_semantic_pipeline()

    elif args.algo == "mapping":
        run_mapping_pipeline()

    elif args.algo == "unseen":
        run_unseen_pipeline(
            unseen_path=args.input,
            model_path=args.model,
            vectorizer_path=args.vec,
        )

    elif args.algo == "unseencold":
        run_unseen_pipeline_cold(
            input_file=args.input,
            model_name=args.model,
            model_path=args.coldmodel,
            n_clusters=18
        )

    else:
        print (f"[MAIN] >>> Error: no such option {args.algo}")


#
#
# Writing the first step, including K-means with anchors
def run_kmeans_pipeline(
        input_file="gin_untagged",
        true_labels="gin_tagged",
        output_file="clustered_output.tsv",
        output_model="model.pkl",
        embeddings_file="embeddings.npy",
        model_name="all-MiniLM-L12-v2",
        anchors=anchor_sentences,
        n_clusters=18
    ):
    # Load sentences from file (1 per line)
    with open(input_file, "r", encoding="utf8") as f:
        sentences = [line.lower().strip() for line in f.readlines() if line.strip()]
    model = SentenceTransformer(model_name)

    print("[K-means] >>> Generating embeddings...")
    embeddings = model.encode(sentences, show_progress_bar=True)
    anchor_embeddings = model.encode(anchor_sentences)

    # Apply K-Means clustering
    print(f"[K-means] >>> Clustering into {n_clusters} categories (anchor-initialized)...")
    kmeans = KMeans(
        n_clusters=n_clusters,
        init=np.array(anchor_embeddings),
        n_init=1,
        random_state=42,
    )
    labels = kmeans.fit_predict(embeddings)
  
    # Shift cluster labels to 1..N
    labels_shifted = [(label + 1) % n_clusters for label in labels] 
                #= [label + 1 for label in labels]
    # Create DataFrame with shifted cluster labels
    df = pd.DataFrame({"Text": sentences, "Cluster": labels_shifted})
    # Display clusters
    for i in range(1, n_clusters + 1):
        print(f"\n[K-means] >>> === Cluster {i} ===")
        print(df[df["Cluster"] == i].head(5)["Text"].to_string(index=False))

    # Write shifted labels with original text to file
    with open(output_file, "w") as f_out:
        for label, sentence in zip(labels_shifted, sentences):
            f_out.write(f"{label}\t{sentence}\n")
    # and also the embeddings of the cold start
    np.save(embeddings_file, embeddings)

    # Save the K-means model.
    if os.path.exists(output_model):
        os.remove(output_model)
    joblib.dump(kmeans, output_model)
    print(f"[K-means] >>> Saved model to {output_model}")  
      
    # Evaluation  
    evaluate_clustering(true_labels, labels_shifted)
    return 0 # Ends OK

#
#
# Definitions of functions relevant to the algorithms executed here
def run_semantic_pipeline():
    # TODO
    return 0 # Ends OK

#
#
# Definitions of functions relevant to the mapping
def run_mapping_pipeline():
    # TODO
    return 0 # Ends OK
  
#
#
# Query the model post-training code
def run_unseen_pipeline(
        unseen_path="gindata/unseen-v2",
        model_path="model.pkl",
        vectorizer_path="vectorizer.pkl"
    ):    
    # Load vectorizer and model
    vectorizer = joblib.load(vectorizer_path)
    clf = joblib.load(model_path)
      
    # Vectorise unseen data using the same vectorizer + reading batch file
    file1=open(unseen_path,"r", encoding="utf8")
    lines = file1.readlines()
    file1.close()
    unseen_data=[]
    for line in lines:
        line=line.lower().strip()
        unseen_data.append(line)
    print (f"[UNSEEN BATCH] >> Read UNSEEN data from {unseen_path} with model {model_path} and vctorizer {vectorizer_path}.")
    
    # Transform with correct vocabulary
    X_unseen = vectorizer.transform(unseen_data)
    # Predict
    predictions = clf.predict(X_unseen)
    # Output
    for text, label in zip(unseen_data, predictions):
        print(f"[{label}] {text}")

# Query the mode post-training (cold)
def run_unseen_pipeline_cold(
    input_file="unseen",
    model_name="all-MiniLM-L12-v2",
    model_path="kmeans.pkl",
    n_clusters=18
):
    # Load the model
    kmeans = joblib.load(model_path)  
    
    # Load sentences from file (1 per line)
    with open(input_file, "r", encoding="utf8") as f:
        sentences = [line.lower().strip() for line in f.readlines() if line.strip()]
    model = SentenceTransformer(model_name)

    print("[K-means] >>> Generating embeddings...")
    embeddings = model.encode(sentences, show_progress_bar=True)
    labels = kmeans.predict(embeddings)

    # Output
    labels_shifted = [(label + 1) % n_clusters for label in labels] 
                #= [label + 1 for label in labels]
    for text, label in zip(sentences, labels_shifted):
        print(f"[{label}] {text}")

    return 0 # Ends OK

# General Utilities
def compute_nmi(true_labels, pred_labels):
    return metrics.normalized_mutual_info_score(true_labels, pred_labels, average_method='arithmetic')

def clustering_accuracy(true_labels, pred_labels):
    cm = confusion_matrix(true_labels, pred_labels)
    row_ind, col_ind = linear_sum_assignment(-cm)  # maximize match
    correct = cm[row_ind, col_ind].sum()
    return correct / cm.sum()

def evaluate_clustering(true_file, pred_labels):
    if not true_file:
        print("[Eval] >>> No true labels provided; skipping evaluation.")
        return 0,0 # Not ideal, but this is no error.
     
    # Load true labels aligned with sentences
    true_labels = []
    with open(true_file, "r", encoding="utf8") as f:
        for raw_line in f:
            line = raw_line.lower().strip()
            #print (line)
            if len(line) > 0:
                label, text = line.split("\t", 1)
                true_labels.append(int(label))

    print(f"[Eval] >>> Read {len(true_labels)} TRUE labels.")
    acc = clustering_accuracy(true_labels, pred_labels)
    nmi = compute_nmi(true_labels, pred_labels)

    print(f"[Eval] >>> Accuracy: {acc:.4f}")
    print(f"[Eval] >>> NMI:      {nmi:.4f}")

    return acc, nmi

if __name__ == "__main__":
    main()
