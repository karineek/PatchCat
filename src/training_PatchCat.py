# COLD: (K-MEANS only) python3 patchCat_clustering.py cold --input gin_untagged --output test_012026_clustered_output.tsv --model all-MiniLM-L12-v2 --truelabels  gin_tagged
# OR  python3 patchCat_clustering.py cold --input gin_untagged --output test_012026_clustered_output.tsv --model all-MiniLM-L12-v2 --truelabels  gin_tagged --ML kmeans++
# OR  python3 patchCat_clustering.py cold --input gin_untagged --output test_012026_clustered_output.tsv --model all-MiniLM-L12-v2 --truelabels  gin_tagged --ML copkmeans
# Cold unseen prediction: (K-MEANS only) python3 patchCat_clustering.py unseencold --input data/unseen --model all-MiniLM-L12-v2 --coldmodel kmeans.pkl  
# Unseen prediction: (Full model)  python3 patchCat_clustering.py unseen --input data/unseen-v2 --vec vectorizer.pkl --model model.pkl

# This Python program contains two separate ML algorithms: 
# KMeans (as in Python libraries) and semantic clustering from the paper
# Zenodo record of this code: https://zenodo.org/records/275118
# https://github.com/Behrouz-Babaki/COP-Kmeans.git --> import this
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

from patchCat_clustering_alts import run_kmeans_pipeline_tfidf
from copkmeans.cop_kmeans import cop_kmeans, cop_predict_using_original_code # Need to be installed, see comments above!


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

# Define ML and CL lists
MUST_LINK = [
(1,16),	(16,31),	(31,46),	(46,61),	(61,76),	(76,91),	(91,106),	(106,121),	(121,136),	(136,151),	(151,166),	(166,181),	(181,196),	(196,211),	(211,226),
(253,269),	(269,285),	(285,301),	(301,317),	(317,333),	(333,349),	(349,365),	(365,381),	(381,397),	(397,413),	(413,429),	(429,445),	(445,461),	(461,477),	(477,493),
(522,539),	(539,556),	(556,573),	(573,590),	(590,607),	(607,624),	(624,641),	(641,658),	(658,675),	(675,692),	(692,709),	(709,726),	(726,743),	(743,760),	(760,777),
(794,816),	(816,838),	(838,860),	(860,882),	(882,904),	(904,926),	(926,948),	(948,970),	(970,992),	(992,1014),	(1014,1036),	(1036,1058),	(1058,1080),	(1080,1102),	(1102,1124),
(1148,1186),	(1186,1224),	(1224,1262),	(1262,1300),	(1300,1338),	(1338,1376),	(1376,1414),	(1414,1452),	(1452,1490),	(1490,1528),	(1528,1566),	(1566,1604),	(1604,1642),	(1642,1680),	(1680,1718),
(1763,1784),	(1784,1805),	(1805,1826),	(1826,1847),	(1847,1868),	(1868,1889),	(1889,1910),	(1910,1931),	(1931,1952),	(1952,1973),	(1973,1994),	(1994,2015),	(2015,2036),	(2036,2057),	(2057,2078),
(2112,2129),	(2129,2146),	(2146,2163),	(2163,2180),	(2180,2197),	(2197,2214),	(2214,2231),	(2231,2248),	(2248,2265),	(2265,2282),	(2282,2299),	(2299,2316),	(2316,2333),	(2333,2350),	(2350,2367),
(2399,2422),	(2422,2445),	(2445,2468),	(2468,2491),	(2491,2514),	(2514,2537),	(2537,2560),	(2560,2583),	(2583,2606),	(2606,2629),	(2629,2652),	(2652,2675),	(2675,2698),	(2698,2721),	(2721,2744),
(2779,2798),	(2798,2817),	(2817,2836),	(2836,2855),	(2855,2874),	(2874,2893),	(2893,2912),	(2912,2931),	(2931,2950),	(2950,2969),	(2969,2988),	(2988,3007),	(3007,3026),	(3026,3045),	(3045,3064),
(3095,3115),	(3115,3135),	(3135,3155),	(3155,3175),	(3175,3195),	(3195,3215),	(3215,3235),	(3235,3255),	(3255,3275),	(3275,3295),	(3295,3315),	(3315,3335),	(3335,3355),	(3355,3375),	(3375,3395),
(3417,3438),	(3438,3459),	(3459,3480),	(3480,3501),	(3501,3522),	(3522,3543),	(3543,3564),	(3564,3585),	(3585,3606),	(3606,3627),	(3627,3648),	(3648,3669),	(3669,3690),	(3690,3711),	(3711,3732),
(3757,3773),	(3773,3789),	(3789,3805),	(3805,3821),	(3821,3837),	(3837,3853),	(3853,3869),	(3869,3885),	(3885,3901),	(3901,3917),	(3917,3933),	(3933,3949),	(3949,3965),	(3965,3981),	(3981,3997),
(4025,4042),	(4042,4059),	(4059,4076),	(4076,4093),	(4093,4110),	(4110,4127),	(4127,4144),	(4144,4161),	(4161,4178),	(4178,4195),	(4195,4212),	(4212,4229),	(4229,4246),	(4246,4263),	(4263,4280),
(4308,4327),	(4327,4346),	(4346,4365),	(4365,4384),	(4384,4403),	(4403,4422),	(4422,4441),	(4441,4460),	(4460,4479),	(4479,4498),	(4498,4517),	(4517,4536),	(4536,4555),	(4555,4574),	(4574,4593),
(4613,4637),	(4637,4661),	(4661,4685),	(4685,4709),	(4709,4733),	(4733,4757),	(4757,4781),	(4781,4805),	(4805,4829),	(4829,4853),	(4853,4877),	(4877,4901),	(4901,4925),	(4925,4949),	(4949,4973),
(4999,5017),	(5017,5035),	(5035,5053),	(5053,5071),	(5071,5089),	(5089,5107),	(5107,5125),	(5125,5143),	(5143,5161),	(5161,5179),	(5179,5197),	(5197,5215),	(5215,5233),	(5233,5251),	(5251,5269),
(5292,5308),	(5308,5324),	(5324,5340),	(5340,5356),	(5356,5372),	(5372,5388),	(5388,5404),	(5404,5420),	(5420,5436),	(5436,5452),	(5452,5468),	(5468,5484),	(5484,5500),	(5500,5516),	(5516,5532),
(5550,5566),	(5566,5582),	(5582,5598),	(5598,5614),	(5614,5630),	(5630,5646),	(5646,5662),	(5662,5678),	(5678,5694),	(5694,5710),	(5710,5726),	(5726,5742),	(5742,5758),	(5758,5774),	(5774,5790)
]

CANNOT_LINK = [
(1,253),	(16,269),	(31,285),	(46,301),	(61,317),	(76,333),	(91,349),	(106,365),	(121,381),	(136,397),	(151,413),	(166,429),	(181,445),	(196,461),	(211,477),	(226,493),
(253,522),	(269,539),	(285,556),	(301,573),	(317,590),	(333,607),	(349,624),	(365,641),	(381,658),	(397,675),	(413,692),	(429,709),	(445,726),	(461,743),	(477,760),	(493,777),
(522,794),	(539,816),	(556,838),	(573,860),	(590,882),	(607,904),	(624,926),	(641,948),	(658,970),	(675,992),	(692,1014),	(709,1036),	(726,1058),	(743,1080),	(760,1102),	(777,1124),
(794,1148),	(816,1186),	(838,1224),	(860,1262),	(882,1300),	(904,1338),	(926,1376),	(948,1414),	(970,1452),	(992,1490),	(1014,1528),	(1036,1566),	(1058,1604),	(1080,1642),	(1102,1680),	(1124,1718),
(1148,1763),	(1186,1784),	(1224,1805),	(1262,1826),	(1300,1847),	(1338,1868),	(1376,1889),	(1414,1910),	(1452,1931),	(1490,1952),	(1528,1973),	(1566,1994),	(1604,2015),	(1642,2036),	(1680,2057),	(1718,2078),
(1763,2112),	(1784,2129),	(1805,2146),	(1826,2163),	(1847,2180),	(1868,2197),	(1889,2214),	(1910,2231),	(1931,2248),	(1952,2265),	(1973,2282),	(1994,2299),	(2015,2316),	(2036,2333),	(2057,2350),	(2078,2367),
(2112,2399),	(2129,2422),	(2146,2445),	(2163,2468),	(2180,2491),	(2197,2514),	(2214,2537),	(2231,2560),	(2248,2583),	(2265,2606),	(2282,2629),	(2299,2652),	(2316,2675),	(2333,2698),	(2350,2721),	(2367,2744),
(2399,2779),	(2422,2798),	(2445,2817),	(2468,2836),	(2491,2855),	(2514,2874),	(2537,2893),	(2560,2912),	(2583,2931),	(2606,2950),	(2629,2969),	(2652,2988),	(2675,3007),	(2698,3026),	(2721,3045),	(2744,3064),
(2779,3095),	(2798,3115),	(2817,3135),	(2836,3155),	(2855,3175),	(2874,3195),	(2893,3215),	(2912,3235),	(2931,3255),	(2950,3275),	(2969,3295),	(2988,3315),	(3007,3335),	(3026,3355),	(3045,3375),	(3064,3395),
(3095,3417),	(3115,3438),	(3135,3459),	(3155,3480),	(3175,3501),	(3195,3522),	(3215,3543),	(3235,3564),	(3255,3585),	(3275,3606),	(3295,3627),	(3315,3648),	(3335,3669),	(3355,3690),	(3375,3711),	(3395,3732),
(3417,3757),	(3438,3773),	(3459,3789),	(3480,3805),	(3501,3821),	(3522,3837),	(3543,3853),	(3564,3869),	(3585,3885),	(3606,3901),	(3627,3917),	(3648,3933),	(3669,3949),	(3690,3965),	(3711,3981),	(3732,3997),
(3757,4025),	(3773,4042),	(3789,4059),	(3805,4076),	(3821,4093),	(3837,4110),	(3853,4127),	(3869,4144),	(3885,4161),	(3901,4178),	(3917,4195),	(3933,4212),	(3949,4229),	(3965,4246),	(3981,4263),	(3997,4280),
(4025,4308),	(4042,4327),	(4059,4346),	(4076,4365),	(4093,4384),	(4110,4403),	(4127,4422),	(4144,4441),	(4161,4460),	(4178,4479),	(4195,4498),	(4212,4517),	(4229,4536),	(4246,4555),	(4263,4574),	(4280,4593),
(4308,4613),	(4327,4637),	(4346,4661),	(4365,4685),	(4384,4709),	(4403,4733),	(4422,4757),	(4441,4781),	(4460,4805),	(4479,4829),	(4498,4853),	(4517,4877),	(4536,4901),	(4555,4925),	(4574,4949),	(4593,4973),
(4613,4999),	(4637,5017),	(4661,5035),	(4685,5053),	(4709,5071),	(4733,5089),	(4757,5107),	(4781,5125),	(4805,5143),	(4829,5161),	(4853,5179),	(4877,5197),	(4901,5215),	(4925,5233),	(4949,5251),	(4973,5269),
(4999,5292),	(5017,5308),	(5035,5324),	(5053,5340),	(5071,5356),	(5089,5372),	(5107,5388),	(5125,5404),	(5143,5420),	(5161,5436),	(5179,5452),	(5197,5468),	(5215,5484),	(5233,5500),	(5251,5516),	(5269,5532),
(5292,5550),	(5308,5566),	(5324,5582),	(5340,5598),	(5356,5614),	(5372,5630),	(5388,5646),	(5404,5662),	(5420,5678),	(5436,5694),	(5452,5710),	(5468,5726),	(5484,5742),	(5500,5758),	(5516,5774),	(5532,5790)
]

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
    cold.add_argument("--ML", default="kmeans")
    cold.add_argument("--vectorizer", default="vectorizer.pkl")

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
    unseen_cold.add_argument("--truelabels", default=None)

    # Parse
    args = parser.parse_args()

    # cold start phase
    if args.algo == "cold":
        if args.model == "tfidf":
            print (f"[MAIN] >>> Running with TF-IDF, (", args.ML, ")")
            run_kmeans_pipeline_tfidf(
				input_file=args.input,
				true_labels=args.truelabels,
				output_file=args.output,
				output_model=args.outmodel,
				vectorizer_model=args.vectorizer,
				embeddings_file=args.embeddings,
				anchors=anchor_sentences, ## Not always used
				n_clusters=18,
				ml=args.ML,
				MUST_LINK=MUST_LINK, # Need to pass from the original code, Not always used
				CANNOT_LINK=CANNOT_LINK, # Need to pass from the original code, Not always used
				evaluate_clustering=evaluate_clustering
            )
        else:
            print (f"[MAIN] >>> Running with embeddings, (", args.embeddings, " and ", args.ML, ")")
            run_kmeans_pipeline(
                input_file=args.input,
                true_labels=args.truelabels,
                output_file=args.output,
                output_model=args.outmodel,
                embeddings_file=args.embeddings,
                model_name=args.model,
                anchors=anchor_sentences,
                n_clusters=18,
                ml=args.ML
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
            true_labels=args.truelabels,
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
        n_clusters=18,
        ml="kmeans"
    ):
    # Load sentences from file (1 per line)
    with open(input_file, "r", encoding="utf8") as f:
        sentences = [line.lower().strip() for line in f.readlines() if line.strip()]
    model = SentenceTransformer(model_name)

    print("[K-means] >>> Generating embeddings...")
    embeddings = model.encode(sentences, show_progress_bar=True)
    anchor_embeddings = model.encode(anchor_sentences)
    # normalising
    if ml == "skmeans":
        eps = 1e-12
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + eps)
        anchor_embeddings = anchor_embeddings / (np.linalg.norm(anchor_embeddings, axis=1, keepdims=True) + eps)
    else:
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        anchor_embeddings = anchor_embeddings / np.linalg.norm(anchor_embeddings, axis=1, keepdims=True)

    # Apply K-Means clustering
    print(f"[K-means] >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Clustering into {n_clusters} categories (anchor-initialized)...")
    if ml == "copkmeans":
        X = np.asarray(embeddings)

        labels, centers = cop_kmeans(
            dataset=X,
            k=n_clusters,
            ml=MUST_LINK,
            cl=CANNOT_LINK,
            init_centers=np.asarray(anchor_embeddings, dtype=float),
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

    else: # General K-MEANS flow
        if ml == "kmeans++":
            print(f"[K-means++] >>> Clustering into {n_clusters} categories...")
            kmeans = KMeans(
                n_clusters=18,
                init="k-means++",
                n_init=10
            )
        else:
            print(f"[K-means wt Anchors] >>> Clustering into {n_clusters} categories (anchor-initialized)...")
            kmeans = KMeans(
                n_clusters=n_clusters,
                init=np.array(anchor_embeddings),
                n_init=1
            )

        labels = kmeans.fit_predict(embeddings)

        if ml == "skmeans":
            centers = kmeans.cluster_centers_
            centers /= np.linalg.norm(centers, axis=1, keepdims=True)
            kmeans.cluster_centers_ = centers


    # Shift cluster labels to 1..N
    labels_shifted = [(label + 1) % n_clusters for label in labels]
    # Create DataFrame with shifted cluster labels
    df = pd.DataFrame({"Text": sentences, "Cluster": labels_shifted})
    # Display clusters
    for i in range(0, n_clusters):
        print(f"\n[K-means] >>> === Cluster {i} ===")
        print(df[df["Cluster"] == i].head(5)["Text"].to_string(index=False))

    # Write shifted labels with original text to file
    with open(output_file, "w") as f_out:
        for label, sentence in zip(labels_shifted, sentences):
            f_out.write(f"{label}\t{sentence}\n")
    # and also the embeddings of the cold start
    np.save(embeddings_file, embeddings)

    # Save the K-means model.
    if ml == "copkmeans":
        # Normalize centers to match normalized embeddings
        eps = 1e-12
        centers = np.asarray(centers, dtype=float)
        centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + eps)

        cop_model = {
            "centers": centers,
            "k": n_clusters,
            "model_name": model_name,
            "eps": eps,
        }

        if os.path.exists(output_model):
            os.remove(output_model)
        joblib.dump(cop_model, output_model)
        print(f"[COP-K-MEANS] >>> Saved centroid model to {output_model}")

    else:
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
    true_labels="gindata/tagged-v2.txt",
    n_clusters=18
):
    try:
        # Load the model
        kmeans = joblib.load(model_path)

        # Load sentences from file (1 per line)
        with open(input_file, "r", encoding="utf8") as f:
            sentences = [line.lower().strip() for line in f.readlines() if line.strip()]
        model = SentenceTransformer(model_name)

        print("[K-means] >>> Generating embeddings...")
        embeddings = model.encode(sentences, show_progress_bar=True)
        labels = kmeans.predict(embeddings)
    except: # We are likely using COPKMEANS!
        copkmeans = joblib.load(model_path)
        centers = np.asarray(copkmeans["centers"], dtype=float)
        eps = float(copkmeans.get("eps", 1e-12))
        X = np.asarray(embeddings, dtype=float)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)
        labels = cop_predict_using_original_code(centers, X)

    # Output
    labels_shifted = [(label + 1) % n_clusters for label in labels]
                #= [label + 1 for label in labels]
    for text, label in zip(sentences, labels_shifted):
        print(f"[{label}] {text}")

    # Evaluation
    evaluate_clustering(true_labels, labels_shifted)

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
                #print(f"Start line {line}")
                label, text = line.split("\t", 1)
                label = label.strip()
                if label.startswith("[") and label.endswith("]"):
                    label = label[1:-1]
                true_labels.append(int(label))
                #print(f"Great! End line {line} with NO ERROR!")

    print(f"[Eval] >>> Read {len(true_labels)} TRUE labels.")
    acc = clustering_accuracy(true_labels, pred_labels)
    nmi = compute_nmi(true_labels, pred_labels)

    print(f"[Eval] >>> Accuracy: {acc:.4f}")
    print(f"[Eval] >>> NMI:      {nmi:.4f}")

    return acc, nmi

if __name__ == "__main__":
    main()
