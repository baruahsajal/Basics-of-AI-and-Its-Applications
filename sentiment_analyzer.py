import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import pickle

nltk.download('stopwords')
nltk.download('punkt')

class SentimentAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = LogisticRegression(max_iter=1000)
        self.stop_words = set(stopwords.words('english'))
    
    def _clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in self.stop_words]
        return ' '.join(tokens)
    
    def train(self, texts, labels):
        cleaned_texts = [self._clean_text(text) for text in texts]
        X = self.vectorizer.fit_transform(cleaned_texts)
        
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        
        print("Accuracy:", accuracy_score(y_test, y_pred))
        
        pickle.dump(self.vectorizer, open('vectorizer.pkl', 'wb'))
        pickle.dump(self.model, open('sentiment_model.pkl', 'wb'))
    
    def predict(self, text):
        cleaned = self._clean_text(text)
        vectorized = self.vectorizer.transform([cleaned])
        prediction = self.model.predict(vectorized)[0]
        return "Positive" if prediction == 1 else "Negative"

if __name__ == '__main__':
    sample_texts = ["AI is incredibly fascinating!", "I am frustrated by the bugs.", "Great learning experience.", "Worst errors ever."]
    sample_labels = [1, 0, 1, 0] 
    
    analyzer = SentimentAnalyzer()
    analyzer.train(sample_texts, sample_labels)
    print("Test Prediction:", analyzer.predict("This algorithm runs perfectly!"))
