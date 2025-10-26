import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

from text_processing import preprocess_series


def map_topics_to_labels(topic_assignments, true_labels):
    """
    Map each topic index to the most frequent true category.
    """
    mapping = {}
    topic_to_labels = defaultdict(list)

    for topic, true_label in zip(topic_assignments, true_labels):
        topic_to_labels[topic].append(true_label)

    for topic, labels in topic_to_labels.items():
        most_common = pd.Series(labels).mode()[0]
        mapping[topic] = most_common

    return mapping


# Load dataset
df = pd.read_csv("./req_data/processed/20news_18828_clean.csv")
texts = preprocess_series(df["text"])
labels = df["category"]

# Load models + vectorizers
lda = joblib.load("models/lda_model.pkl")
count_vectorizer = joblib.load("models/lda_vectorizer.pkl")

nmf = joblib.load("models/nmf_model.pkl")
tfidf_vectorizer = joblib.load("models/nmf_vectorizer.pkl")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
)

# ----- LDA -----
X_counts_test = count_vectorizer.transform(X_test)
doc_topic_lda = lda.transform(X_counts_test)
lda_topics = np.argmax(doc_topic_lda, axis=1)

# Map topics → categories
lda_mapping = map_topics_to_labels(lda_topics, y_test)
lda_preds = [lda_mapping[t] for t in lda_topics]

print("\n🔹 LDA Evaluation:")
print("Accuracy:", accuracy_score(y_test, lda_preds))
print("\nClassification Report:\n", classification_report(y_test, lda_preds))

cm_lda = confusion_matrix(y_test, lda_preds, labels=sorted(df["category"].unique()))
plt.figure(figsize=(12, 10))
sns.heatmap(cm_lda, cmap="Blues", xticklabels=sorted(df["category"].unique()),
            yticklabels=sorted(df["category"].unique()), annot=False, fmt="d")
plt.title("LDA Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("lda_confusion_matrix.png")
plt.show()

# ----- NMF -----
X_tfidf_test = tfidf_vectorizer.transform(X_test)
doc_topic_nmf = nmf.transform(X_tfidf_test)
nmf_topics = np.argmax(doc_topic_nmf, axis=1)

# Map topics → categories
nmf_mapping = map_topics_to_labels(nmf_topics, y_test)
nmf_preds = [nmf_mapping[t] for t in nmf_topics]

print("\n🔹 NMF Evaluation:")
print("Accuracy:", accuracy_score(y_test, nmf_preds))
print("\nClassification Report:\n", classification_report(y_test, nmf_preds))

cm_nmf = confusion_matrix(y_test, nmf_preds, labels=sorted(df["category"].unique()))
plt.figure(figsize=(12, 10))
sns.heatmap(cm_nmf, cmap="Greens", xticklabels=sorted(df["category"].unique()),
            yticklabels=sorted(df["category"].unique()), annot=False, fmt="d")
plt.title("NMF Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("nmf_confusion_matrix.png")
plt.show()