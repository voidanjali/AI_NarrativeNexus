import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# Initialize NLTK components
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def nlp_preprocess(text):
    """Tokenize, clean, remove stopwords, lemmatize"""
    if not isinstance(text, str):
        return ""
    
    tokens = nltk.word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

def preprocess_series(X):
    """Apply nlp_preprocess to a list/Series"""
    return [nlp_preprocess(t) for t in X]


df = pd.read_csv("req_data/processed/20news_18828_clean.csv")
X = df["text"]
y = df["category"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline([
    ("preprocess", FunctionTransformer(preprocess_series)),
    ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=200))
])


pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)
print(" Accuracy:", accuracy_score(y_test, y_pred))
print("\n Classification Report:\n", classification_report(y_test, y_pred))


os.makedirs("models", exist_ok=True)
model_path = os.path.join("models", "topic_classifier.pkl")
joblib.dump(pipeline, model_path)
print(f" Model saved at: {model_path}")
