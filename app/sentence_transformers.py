from transformers import BertTokenizer, BertForSequenceClassification, AdamW    
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, InputExample, SentencesDataset, losses, models
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

class SentenceTransformersModel:
    def __init__(self):
        word_embedding_model = models.Transformer('model-download/bert-base-uncased', max_seq_length=256)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=256, activation_function=nn.Tanh())
        # super().__init__(*args, **kwargs)
        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])
        self.model.trainable = True
        # self.parameters = self.model.parameters()

    def predict(self, sentence1, sentence2):
        embeddings1 = self.model.encode([sentence1])
        embeddings2 = self.model.encode([sentence2])
        
        similarity = cosine_similarity(embeddings1, embeddings2)
        return similarity[0][0]

    def train_model(self, sentence_pairs):
        self.model.train()

        num_epochs = 3
        loss_function = losses.CosineSimilarityLoss(model=self.model)

        for epoch in range(num_epochs):
            total_loss = 0

            for pair in sentence_pairs:
                sentence1 = pair.sentence1
                sentence2 = pair.sentence2
                cosine_similarity_value = pair.cosine_similarity

                # Encode sentences and calculate similarity
                embeddings1 = self.model.encode([sentence1])
                embeddings2 = self.model.encode([sentence2])
                similarity = cosine_similarity(embeddings1, embeddings2)

                # Calculate loss and perform backpropagation
                loss = loss_function(similarity, cosine_similarity_value)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(sentence_pairs)
            print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')

        # Save the trained model
        self.model.save('trained_sentence_transformers_model')