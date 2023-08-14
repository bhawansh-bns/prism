# from transformers import AdamW
# from sentence_transformers import SentenceTransformer, losses
# from torch.utils.data import DataLoader, Dataset
# import torch
# from app import db
# from sklearn.metrics.pairwise import cosine_similarity
# from app.models import SentencePair
from transformers import BertTokenizer, BertForSequenceClassification, AdamW    
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, InputExample, SentencesDataset, losses
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import SentencePair
import torch.nn as nn
import torch.optim as optim
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
from sklearn.model_selection import train_test_split
from app import db

class CustomDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

class BERTModel:
    def __init__(self, model_path='./model-download/bert-base-uncased'):
        self.model = SentenceTransformer(model_path)

    def predict(self, sentence1, sentence2):
        embeddings1 = self.model.encode([sentence1])
        embeddings2 = self.model.encode([sentence2])
        
        similarity = cosine_similarity(embeddings1, embeddings2)
        return similarity[0][0]

    def train_model(self):
        sentence_pairs = db.session.query(SentencePair).all()
        sentences1 = [pair.sentence1 for pair in sentence_pairs]
        sentences2 = [pair.sentence2 for pair in sentence_pairs]
        cosine_similarities = [pair.cosine_similarity for pair in sentence_pairs]

        train_data = CustomDataset(texts=sentences1, labels=cosine_similarities)
        train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True)

        # Define loss function and optimizer
        loss_function = losses.CosineSimilarityLoss(model=self.model)
        optimizer = AdamW(self.model.parameters(), lr=2e-5)

        # Training loop
        num_epochs = 3
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0
            for batch_texts, batch_labels in train_dataloader:
                embeddings = self.model.encode(batch_texts)
                batch_similarity = cosine_similarity(embeddings, embeddings)
                batch_loss = loss_function(batch_similarity, batch_labels)
                
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                train_loss += batch_loss.item()

            avg_train_loss = train_loss / len(train_dataloader)
            print(f'Epoch {epoch+1}/{num_epochs}, Average Training Loss: {avg_train_loss:.4f}')

        # Save the trained model
        self.model.save('trained_model')
