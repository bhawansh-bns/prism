import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from app.models import SentencePair
from app import db

class CustomDataset(Dataset):
    def __init__(self, sentences1, sentences2, similarities):
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.similarities = similarities

    def __len__(self):
        return len(self.similarities)

    def __getitem__(self, idx):
        return self.sentences1[idx], self.sentences2[idx], self.similarities[idx]

class CosineSimilarityModel(nn.Module):
    def __init__(self, embedding_model):
        super(CosineSimilarityModel, self).__init__()
        self.embedding_model = embedding_model

    def forward(self, sentences1, sentences2):
        embeddings1 = self.embedding_model.encode(sentences1)
        embeddings2 = self.embedding_model.encode(sentences2)
        similarities = torch.nn.functional.cosine_similarity(embeddings1, embeddings2)
        return similarities

def train_cosine_similarity_model(embedding_model):
    sentence_pairs = db.session.query(SentencePair).all()
    sentences1 = [pair.sentence1 for pair in sentence_pairs]
    sentences2 = [pair.sentence2 for pair in sentence_pairs]
    cosine_similarities = [pair.cosine_similarity for pair in sentence_pairs]

    train_sentences1, val_sentences1, train_sentences2, val_sentences2, train_similarities, val_similarities = train_test_split(
        sentences1, sentences2, cosine_similarities, test_size=0.2, random_state=42
    )

    train_dataset = CustomDataset(train_sentences1, train_sentences2, train_similarities)
    val_dataset = CustomDataset(val_sentences1, val_sentences2, val_similarities)

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16)

    model = CosineSimilarityModel(embedding_model)
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    loss_fn = nn.MSELoss()

    num_epochs = 3
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_sentences1, batch_sentences2, batch_similarities in train_dataloader:
            optimizer.zero_grad()
            batch_predictions = model(batch_sentences1, batch_sentences2)
            batch_loss = loss_fn(batch_predictions, batch_similarities)
            batch_loss.backward()
            optimizer.step()
            train_loss += batch_loss.item()

        avg_train_loss = train_loss / len(train_dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Average Training Loss: {avg_train_loss:.4f}')

    # Save the trained model
    torch.save(model.state_dict(), 'trained_cosine_model.pth')
