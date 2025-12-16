import sys
from joblib import load

# Load model and vectorizer
vectorizer = load("/home/carol/Desktop/karineFOrk/gin-llm/clustering/running-model/vectorizer.pkl")
clf = load("/home/carol/Desktop/karineFOrk/gin-llm/clustering/running-model/model.pkl")

def classify(text: str) -> str:
    clean_text = text.lower().strip()
    X = vectorizer.transform([clean_text])
    return clf.predict(X)[0]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_text.py 'your text here'")
        sys.exit(1)

    input_text = sys.argv[1]
    label = classify(input_text)
    print(f"[{label}]\t{input_text}")
