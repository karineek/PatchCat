# Model

This is a ready model to use within gin

```
ubuntu@fuzzing-05:~/gin/git-backup/gin-llm/clustering/running-model$ python3 unseen-retrives-batch.py "no diff"
[1]	no diff
ubuntu@fuzzing-05:~/gin/git-backup/gin-llm/clustering/running-model$ python3 unseen-retrives-batch.py "just adding comments"
[2]	just adding comments
ubuntu@fuzzing-05:~/gin/git-backup/gin-llm/clustering/running-model$ python3 unseen-retrives-batch.py "just added try and catch"
[9]	just added try and catch
```

This is the running model used in the PatchCat ASE NIER 2025 paper.

_Even-Mendoza, K., Brownlee, A., Geiger, A., Hanna, C., Petke, J., Sarro, F., & Sobania, D. (2025). LLM-Guided Genetic Improvement: Envisioning Semantic Aware Automated Software Evolution. In New Ideas and Emerging Results Track, 40th IEEE/ACM International Conference on Automated Software Engineering, ASE 2025: ASE 2025 NIER
_

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
