from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from utils import read_train_split

def load_dataset(merge_labels=False):
    train_data, dev_data = read_train_split(merge_labels=merge_labels)
    texts_train = train_data['text'].to_list()
    labels_train = train_data['label'].to_list()
    texts_val = dev_data['text'].to_list()
    labels_val = dev_data['label'].to_list()
    return {"X_train": texts_train, "y_train": labels_train, "X_test": texts_val, "y_test": labels_val}

class Baseline:
    def __init__(self, model, vectorizer_type="bow"):
        self.model = model
        self.dataset = None
        if vectorizer_type == "bow":
            self.vectorizer = CountVectorizer()
        elif vectorizer_type == "tfidf":
            self.vectorizer = TfidfVectorizer()
        else: 
            raise ValueError("Vectorizer type not supported")
        
    def read_data(self, merge_labels=False):
        self.dataset = load_dataset(merge_labels=merge_labels)
        
    def encoding(self):
        # Convert text to BoW or TF-IDF features
        encoding = self.vectorizer
        self.dataset['X_train'] = encoding.fit_transform(self.dataset['X_train'])
        self.dataset['X_test'] = encoding.transform(self.dataset['X_test'])
        
        # Scale the data
        scaler = StandardScaler(with_mean=False)  # Keep 'with_mean=False' for sparse matrices
        self.dataset['X_train'] = scaler.fit_transform(self.dataset['X_train']) 
        self.dataset['X_test'] = scaler.transform(self.dataset['X_test'])  # Scale X_test too!

    def train(self):
        self.model.fit(self.dataset['X_train'], self.dataset['y_train'])
    
    def evaluate(self):
        pred = self.model.predict(self.dataset['X_test'])
        
        accuracy = accuracy_score(self.dataset['y_test'], pred)
        f1 = f1_score(self.dataset['y_test'], pred)
        
        return {
            'accuracy': accuracy,
            'f1_score': f1
        }
        
if __name__ == "__main__":
    
    # Test two baselines
    vectorizer_types = ["bow", "tfidf"]
    for vectorizer_type in vectorizer_types:
        # Initialize model
        model = LogisticRegression(max_iter=500)
        
        # Initialize baseline
        baseline = Baseline(model=model, vectorizer_type=vectorizer_type)
        
        # Load data
        baseline.read_data(merge_labels=True)
        
        # Encode data
        baseline.encoding()
        
        # Train and evaluate
        baseline.train()
        results = baseline.evaluate()
        print(f"Baseline {vectorizer_type.capitalize()} with Logistic Regression:")
        print("Accuracy:", results['accuracy'])
        print("F1 Score:", results['f1_score'])
        print("-" * 50)