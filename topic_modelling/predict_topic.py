import joblib
from text_processing import preprocess_series
import numpy as np

# Load models
lda = joblib.load("models/lda_model.pkl")
count_vectorizer = joblib.load("models/lda_vectorizer.pkl")

nmf = joblib.load("models/nmf_model.pkl")
tfidf_vectorizer = joblib.load("models/nmf_vectorizer.pkl")

examples = [
    "The new graphics card from NVIDIA has amazing performance for 3D rendering.",
    "God does not exist, and religion is just a human creation.",
    "NASA discovered water on Mars, confirming planetary research findings."
]


examples_clean = preprocess_series(examples)


lda_features = count_vectorizer.transform(examples_clean)
lda_topics = lda.transform(lda_features)
print("\n🔹 LDA Predictions:")
for text, topic in zip(examples, np.argmax(lda_topics, axis=1)):
    print(f"\nText: {text}\nPredicted Topic: {topic}")


nmf_features = tfidf_vectorizer.transform(examples_clean)
nmf_topics = nmf.transform(nmf_features)
print("\n🔹 NMF Predictions:")
for text, topic in zip(examples, np.argmax(nmf_topics, axis=1)):
    print(f"\nText: {text}\nPredicted Topic: {topic}")