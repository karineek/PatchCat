from joblib import load

v = load("vectorizer.pkl")
m = load("model.pkl")

# Load vectorizer and model
vectorizer = v
clf = m

unseen_data=[]
line="i apologize, but there is no java diff to summarize."
line=line.lower().strip()
unseen_data.append(line)

# Transform with correct vocabulary
X_unseen = vectorizer.transform(unseen_data)

# Predict
predictions = clf.predict(X_unseen)

# Output
for text, label in zip(unseen_data, predictions):
    print(f"[{label}]\t{text}")
