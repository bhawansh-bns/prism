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
    def __init__(self):
        self.model = SentenceTransformer()
        self.model.trainable = True

    def predict(self, sentence1, sentence2):
        embeddings1 = self.model.encode([sentence1])
        embeddings2 = self.model.encode([sentence2])
        
        similarity = cosine_similarity(embeddings1, embeddings2)
        return similarity[0][0]

    def train_model(self):
        # Create a dataset of sentence pairs and their cosine similarity values
        sentence_pairs = [
            ('Set the temperature to 72 degrees', 'Adjust thermostat to 72 degrees', 0.8340750932693481),
            ('Turn on the lamp in the living room', 'Switch on the living room lamp', 0.8756259679794312),
            ('Lock the back door', 'Secure the rear entrance', 0.8282058835029602),
        ]

        # Create a DataLoader object for the dataset
        dataloader = torch.utils.data.DataLoader(sentence_pairs, batch_size=16)

        # Define a loss function and an optimizer
        loss_function = losses.CosineSimilarityLoss(model=model)
        optimizer = AdamW(model.parameters(), lr=2e-5)

        # Train the model
        for epoch in range(3):
            model.train()
            for batch in dataloader:
                sentences, labels = batch
                embeddings = model.encode(sentences)
                batch_loss = loss_function(embeddings, labels)
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

        # Save the trained model
        model.save('trained_model')

