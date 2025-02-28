import logging
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from utils import read_train_split
import numpy as np
import random
import pandas as pd

# Configure logging
logging.basicConfig(filename='./baseline/baseline.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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
        self.original_dataset = None
        self.vectorizer_type = vectorizer_type
        
        if vectorizer_type == "bow":
            self.vectorizer = CountVectorizer()
        elif vectorizer_type == "tfidf":
            self.vectorizer = TfidfVectorizer()
        else: 
            raise ValueError("Vectorizer type not supported")
        
    def read_data(self, merge_labels=False):
        self.dataset = load_dataset(merge_labels=merge_labels)
        self.original_dataset = self.dataset.copy()
        
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
            'pred': pred,
            'accuracy': accuracy,
            'f1_score': f1
        }

    def get_feature_names_out(self):
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        # Get the top contributing features
        feature_importance = np.abs(self.model.coef_).mean(axis=0)
        top_features_idx = np.argsort(feature_importance)[-10:]
        top_features = feature_names[top_features_idx]

        return top_features
        
    def identify_misclassified_examples(self, pred):

        misclassified_idx = np.where(pred != self.dataset['y_test'])[0]
        
        misclassified_text, true_label, predicted_label = None, None, None
        
        if len(misclassified_idx) > 0:
            # misclassified_example_idx = misclassified_idx[random.randint(0, len(misclassified_idx)-1)]
            # misclassified_text = self.original_dataset["X_test"][misclassified_example_idx]
            # true_label = int(self.dataset["y_test"][misclassified_example_idx])
            # predicted_label = int(pred[misclassified_example_idx])
            # misclassified_example_idx = misclassified_idx[random.randint(0, len(misclassified_idx)-1)]
            X_test = self.original_dataset["X_test"]
            misclassified_text = [X_test[i] for i in misclassified_idx]
            true_label = [int(self.dataset["y_test"][i]) for i in misclassified_idx]
            predicted_label = [int(pred[i]) for i in misclassified_idx]
            mis_ex = {"text": misclassified_text, "true label": true_label, "predicted label": predicted_label}
            misclassified_df = pd.DataFrame(mis_ex)
            misclassified_df.to_csv(f"./baseline/{self.vectorizer_type}_baseline_misclassified_examples.csv", index=False)

        return {
            "Misclassified Example": misclassified_text,
            "True Label": true_label,
            "Predicted Label": predicted_label
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
        print(f"Accuracy: {results['accuracy']}")
        print(f"F1 Score: {results['f1_score']}")
        print("-" * 50)
        
        ########################################

        # Get feature names
        feature_names = baseline.vectorizer.get_feature_names_out()
        # Get the top contributing features
        feature_importance = np.abs(model.coef_).mean(axis=0)
        top_features_idx = np.argsort(feature_importance)[-10:]
        top_features = feature_names[top_features_idx]
        print(f"Top 10 features for {vectorizer_type.capitalize()}:")
        print(top_features)
        print("-" * 50)
        
        ### Identify misclassified samples
        pred = results['pred']
        mis_ex = baseline.identify_misclassified_examples(pred)
        
        baseline.identify_misclassified_examples(pred)
        print(f"Misclassified Example for {vectorizer_type.capitalize()}:")
        print("-" * 50)
        print("\n")