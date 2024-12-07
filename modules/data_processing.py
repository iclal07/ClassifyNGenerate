import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

class TextProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        words = [word for word in text.split() if word not in self.stop_words]
        words = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(words)

    def preprocess_data(self, train_file, test_file):
        train_df = pd.read_parquet(train_file)
        test_df = pd.read_parquet(test_file)
        train_df['cleaned_text'] = train_df['text'].apply(self.clean_text)
        test_df['cleaned_text'] = test_df['text'].apply(self.clean_text)
        return train_df, test_df

