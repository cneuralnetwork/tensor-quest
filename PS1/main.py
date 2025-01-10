import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import ssl
import warnings
warnings.filterwarnings('ignore')

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

for package in ['punkt','punkt_tab', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']:
    try:
        nltk.download(package, quiet=True)
    except Exception as e:
        print(f"Error downloading {package}: {str(e)}")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    try:
        text = str(text).lower()
        text = re.sub(r'£|\$|€', ' money ', text)
        text = re.sub(r'\d+%', ' percent ', text)
        text = re.sub(r'\b\d+\b', ' number ', text)
        
        spam_patterns = {
            'txt': 'text',
            'ur': 'your',
            'u': 'you',
            'yr': 'year',
            'won': 'win',
            'winning': 'win',
            'winner': 'win',
            'free': 'free_gift',
            'prize': 'reward',
            'urgent': 'important',
            'congrat': 'congratulation',
            'click': 'link',
            'offer': 'deal'
        }
        for pattern, replacement in spam_patterns.items():
            text = re.sub(fr'\b{pattern}\b', replacement, text)
        
        text = re.sub(r'http\S+|www\S+', 'url', text)
        text = re.sub(r'[^a-zA-Z\s!?.]', ' ', text)
        text = ' '.join(text.split())
        
        try:
            words = word_tokenize(text)
            stop_words = set(stopwords.words('english')) - {'no', 'not', 'free', 'won', 'win'}
            words = [w for w in words if w not in stop_words]
            lemmatizer = WordNetLemmatizer()
            words = [lemmatizer.lemmatize(w) for w in words]
            bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
            words.extend(bigrams)
        except Exception as e:
            print(f"NLTK processing failed: {str(e)}")
        
        return ' '.join(words)
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        return text

def create_features(data):
    tfidf = TfidfVectorizer(
        max_features=6000,
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    
    X_tfidf = tfidf.fit_transform(data['processed_message'])
    
    X_length = np.array([
        [
            len(text),
            text.count('!'),
            text.count('?'),
            text.count('.'),
            len(text.split()),
            sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0,
        ]
        for text in data['message']
    ])
    
    X_combined = np.hstack([X_tfidf.toarray(), X_length])
    
    return X_combined, tfidf

class SpamDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class EnhancedSpamClassifier(nn.Module):
    def __init__(self, input_size):
        super(EnhancedSpamClassifier, self).__init__()
        
        self.deep_track = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.shallow_track = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.combined = nn.Sequential(
            nn.Linear(384, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        deep_features = self.deep_track(x)
        shallow_features = self.shallow_track(x)
        combined = torch.cat([deep_features, shallow_features], dim=1)
        return self.combined(combined)

def train_model(model, train_loader, criterion, optimizer, scheduler, device, num_epochs=100):
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            l1_lambda = 1e-5
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss += l1_lambda * l1_norm
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
        
        scheduler.step(epoch_loss)

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    return accuracy, all_preds, all_labels

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading data...")
    data = pd.read_csv("/home/deeponh/Tensor Quest/PS1/archive/Spam_SMS.csv")
    data.columns = ["type", "message"]
    
    print("Preprocessing messages...")
    data['processed_message'] = data['message'].apply(preprocess_text)
    
    print("\nCreating features...")
    X, vectorizer = create_features(data)
    y = LabelEncoder().fit_transform(data['type'])
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\nTraining fold {fold+1}/5...")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        y_train = torch.LongTensor(y_train)
        y_test = torch.LongTensor(y_test)
        
        train_dataset = SpamDataset(X_train, y_train)
        test_dataset = SpamDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        model = EnhancedSpamClassifier(X_train.shape[1]).to(device)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0]).to(device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)
        
        train_model(model, train_loader, criterion, optimizer, scheduler, device)
        fold_accuracy, preds, labels = evaluate_model(model, test_loader, device)
        accuracies.append(fold_accuracy)
        print(f'Fold {fold+1} Accuracy: {fold_accuracy:.2f}%')
    
    print(f'\nAverage Accuracy across folds: {np.mean(accuracies):.2f}%')
    print(f'Standard Deviation: {np.std(accuracies):.2f}%')
    
    torch.save(model.state_dict(), 'spam_classifier.pth')
    
    return model, vectorizer

def predict_message(message, model, vectorizer, device):
    processed_message = preprocess_text(message)
    feature_vector = create_features(pd.DataFrame({'message': [message], 'processed_message': [processed_message]}))[0]
    message_tensor = torch.FloatTensor(feature_vector).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(message_tensor)
        _, predicted = torch.max(output.data, 1)
    
    return "spam" if predicted.item() == 1 else "ham"

if __name__ == "__main__":
    model, vectorizer = main()
    
    test_messages = [
        "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward!",
        "Hey, what time are we meeting for lunch tomorrow?",
        "CONGRATULATIONS! You've won a FREE iPhone! Click here to claim your prize now!"
    ]
    
    print("\nTesting model with example messages:")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for msg in test_messages:
        prediction = predict_message(msg, model, vectorizer, device)
        print(f'\nMessage: {msg[:50]}...\nPrediction: {prediction}')
