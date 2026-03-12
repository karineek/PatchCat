import os
import joblib
import numpy as np
import pandas as pd

from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

from copkmeans.cop_kmeans import cop_kmeans, cop_predict_using_original_code # Need to be installed, see comments above!

# Acknowledgements
# Date: 2026-01-30
# Notes: Implementation developed by the author with
#        iterative design and refactoring assistance
#        from ChatGPT (OpenAI).
# We gave the original K-means with embeddings to
# openAI, and it produced this code to test with
# TF-IDF.


# You already have these in your file:
# - anchor_sentences (or passed as anchors)
# - MUST_LINK, CANNOT_LINK
# - evaluate_clustering(true_labels_path, pred_labels)
# - COPKMeans (if ml == "copkmeans")

#
#
# Writing the first step, including K-means with anchors
def run_kmeans_pipeline_tfidf(
        input_file="gin_untagged",
        true_labels="gin_tagged",
        output_file="clustered_output.tsv",
        output_model="model.pkl",
        vectorizer_model="vectorizer.pkl",
        embeddings_file="tfidf_embeddings.npz",
        anchors=None,
        n_clusters=18,
        ml="kmeans",
        MUST_LINK=None, # Need to pass form the original code
        CANNOT_LINK=None, # Need to pass form the original code
	evaluate_clustering=None, # Need to pass form the original code (this is a function, but this is okay in Python)
 	# Some parameters to test with
        tfidf_mode="char_wb",
        word_ngram_range=(1, 2),
        char_ngram_range=(3, 5),
        min_df=2,
        max_df=0.95,
        use_svd=False,
        svd_components=256,
        random_state=42,
    ):
    # -----------------------
    # Load sentences
    # -----------------------
    with open(input_file, "r", encoding="utf8") as f:
        sentences = [line.lower().strip() for line in f.readlines() if line.strip()]

    # -----------------------
    # Build TF-IDF features
    # -----------------------
    if tfidf_mode == "word":
        vec_word = TfidfVectorizer(
            analyzer="word",
            ngram_range=word_ngram_range,
            min_df=min_df,
            max_df=max_df
        )
        X = vec_word.fit_transform(sentences)
        A = vec_word.transform(anchors) if anchors is not None else None

        vectorizer_bundle = {"mode": "word", "word": vec_word}

    elif tfidf_mode == "char_wb":
        vec_char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=char_ngram_range,
            min_df=min_df,
            max_df=max_df
        )
        X = vec_char.fit_transform(sentences)
        A = vec_char.transform(anchors) if anchors is not None else None

        vectorizer_bundle = {"mode": "char_wb", "char": vec_char}

    elif tfidf_mode == "both":
        # Two TF-IDF spaces concatenated: word + char_wb
        vec_word = TfidfVectorizer(
            analyzer="word",
            ngram_range=word_ngram_range,
            min_df=min_df,
            max_df=max_df
        )
        vec_char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=char_ngram_range,
            min_df=min_df,
            max_df=max_df
        )

        Xw = vec_word.fit_transform(sentences)
        Xc = vec_char.fit_transform(sentences)
        X = sparse.hstack([Xw, Xc], format="csr")

        if anchors is not None:
            Aw = vec_word.transform(anchors)
            Ac = vec_char.transform(anchors)
            A = sparse.hstack([Aw, Ac], format="csr")
        else:
            A = None

        vectorizer_bundle = {"mode": "both", "word": vec_word, "char": vec_char}

    else:
        raise ValueError(f"Unknown tfidf_mode: {tfidf_mode}")

    # L2 normalize rows (cosine-friendly)
    X = normalize(X, norm="l2", axis=1)
    if A is not None:
        A = normalize(A, norm="l2", axis=1)

    # Save vectorizer bundle (always)
    if os.path.exists(vectorizer_model):
        os.remove(vectorizer_model)
    joblib.dump(vectorizer_bundle, vectorizer_model)

    # -----------------------
    # Optional SVD compression
    # -----------------------
    # Needed if COPKMeans expects dense, or if X is huge and you want speed/memory relief.
    svd = None
    if use_svd:
        svd = TruncatedSVD(n_components=svd_components, random_state=random_state)
        X_dense = svd.fit_transform(X)  # dense (N x k)
        X_dense = X_dense / (np.linalg.norm(X_dense, axis=1, keepdims=True) + 1e-12)

        if A is not None:
            A_dense = svd.transform(A)
            A_dense = A_dense / (np.linalg.norm(A_dense, axis=1, keepdims=True) + 1e-12)
        else:
            A_dense = None

        # Store SVD into the same bundle file
        vectorizer_bundle_with_svd = dict(vectorizer_bundle)
        vectorizer_bundle_with_svd["svd"] = svd
        joblib.dump(vectorizer_bundle_with_svd, vectorizer_model)

        feat_for_cluster = X_dense
        anchor_for_init = A_dense
    else:
        feat_for_cluster = X              # sparse
        anchor_for_init = (A.toarray() if A is not None else None)  # init must be dense

    # -----------------------
    # Clustering
    # -----------------------
    print(f"[TF-IDF] >>> Clustering into {n_clusters} categories... mode={ml}")
    model_to_save = None
    if ml == "copkmeans":
        #X = np.asarray(feat_for_cluster)
        X = feat_for_cluster.toarray().astype(float)
        #A = anchor_for_init.toarray().astype(float)

        labels, centers = cop_kmeans(
            dataset=X,
            k=n_clusters,
            ml=MUST_LINK,
            cl=CANNOT_LINK,
            init_centers=np.asarray(anchor_for_init, dtype=float),
            initialization="kmpp",   # ignored when init_centers is not None (fine to keep)
        )

        if labels is None:
            raise ValueError("COP-KMeans failed: constraints made the assignment infeasible (returned None).")

        labels = np.asarray(labels, dtype=int)

        # sanity checks
        if labels.shape[0] != X.shape[0]:
            raise ValueError(f"COP-KMeans returned {labels.shape[0]} labels for {X.shape[0]} samples.")

        if (labels < 0).any():
            missing = np.where(labels < 0)[0][:20]
            raise ValueError(f"COP-KMeans left some points unassigned (label=-1), examples: {missing}")

        if len(np.unique(labels)) != n_clusters:
            # not necessarily fatal, but it explains empty clusters in printing/eval
            print(f"[COP-KMEANS] >>> Warning: got {len(np.unique(labels))} non-empty clusters out of {n_clusters}.")


    else:
        if ml == "kmeans++":
            print(f"[TF-IDF K-Means++] >>> init=k-means++")
            kmeans = KMeans(
                n_clusters=n_clusters,
                init="k-means++",
                n_init=10,
                random_state=random_state
            )
        else:
            print(f"[TF-IDF Anchored K-Means] >>> init=anchors")
            if anchor_for_init is None:
                raise ValueError("anchors provided but anchor vectors are None (check anchors / vectorizer).")
            kmeans = KMeans(
                n_clusters=n_clusters,
                init=np.array(anchor_for_init),
                n_init=1,
                random_state=random_state
            )

        labels = kmeans.fit_predict(feat_for_cluster)
        model_to_save = kmeans

    # -----------------------
    # Shift labels (kept like your code, though it's a bit odd)
    # -----------------------
    labels_shifted = [(int(label) + 1) % n_clusters for label in labels]

    # -----------------------
    # Display cluster heads
    # -----------------------
    df = pd.DataFrame({"Text": sentences, "Cluster": labels_shifted})
    for i in range(0, n_clusters):
        print(f"\n[TF-IDF] >>> === Cluster {i} ===")
        print(df[df["Cluster"] == i].head(5)["Text"].to_string(index=False))

    # -----------------------
    # Write output
    # -----------------------
    with open(output_file, "w", encoding="utf8") as f_out:
        for label, sentence in zip(labels_shifted, sentences):
            f_out.write(f"{label}\t{sentence}\n")

    # -----------------------
    # Save features
    # -----------------------
    if embeddings_file:
        if ml == "copkmeans":
            np.save(embeddings_file, X)
        elif use_svd:
            # Dense saved as .npy
            dense_path = embeddings_file.replace(".npz", ".npy")
            np.save(dense_path, feat_for_cluster)
        else:
            # Sparse saved as .npz
            sparse.save_npz(embeddings_file, X)

    # -----------------------
    # Save model
    # -----------------------
    if ml == "copkmeans":
        # Normalize centers to match normalized embeddings
        eps = 1e-12
        centers = np.asarray(centers, dtype=float)
        centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + eps)

        cop_model = {
            "centers": centers,
            "k": n_clusters,
            "model_name": "tfidf",
            "eps": eps,
        }

        if os.path.exists(output_model):
            os.remove(output_model)
        joblib.dump(cop_model, output_model)
        print(f"[TF-IDF-COP-K-MEANS] >>> Saved centroid model to {output_model}")
    else:
        if os.path.exists(output_model):
            os.remove(output_model)
        joblib.dump(model_to_save, output_model)
        print(f"[TF-IDF] >>> Saved clustering model to {output_model}")

    print(f"[TF-IDF] >>> Saved vectorizer bundle to {vectorizer_model}")

    # -----------------------
    # Eval
    # -----------------------
    evaluate_clustering(true_labels, labels_shifted)
    return 0
