import pickle

CLASS_LABELS = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}

class Predictor:
    def __init__(self, model_path, vectorizer_path):
        with open(model_path, 'rb') as model_file:
            self.model = pickle.load(model_file)
        with open(vectorizer_path, 'rb') as vec_file:
            self.vectorizer = pickle.load(vec_file)

    def predict(self, text, clean_text_func):
        cleaned_text = clean_text_func(text)
        text_tfidf = self.vectorizer.transform([cleaned_text])
        prediction = self.model.predict(text_tfidf)[0]
        return CLASS_LABELS[prediction]
