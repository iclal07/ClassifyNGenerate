import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class ModelTrainer:
    def __init__(self, model_path, vectorizer_path):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path

    def train_and_save_model(self, train_df):
        X = train_df['cleaned_text']
        y = train_df['label']
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_val_tfidf = vectorizer.transform(X_val)

        model = LogisticRegression(max_iter=500, random_state=42)
        model.fit(X_train_tfidf, y_train)

        print("Validation Performance:\n", classification_report(y_val, model.predict(X_val_tfidf)))

        with open(self.model_path, 'wb') as model_file:
            pickle.dump(model, model_file)
        with open(self.vectorizer_path, 'wb') as vec_file:
            pickle.dump(vectorizer, vec_file)
