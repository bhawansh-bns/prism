from transformers import BertTokenizer, BertForSequenceClassification, AdamW    
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer   

class BERTModel:
    def __init__(self, model_path='./model-download/bert-base-uncased'):
        self.model = SentenceTransformer(model_path)
        self.tokenizer = None 
    
    def predict(self, sentence1, sentence2):
        embeddings1 = self.model.encode([sentence1])
        embeddings2 = self.model.encode([sentence2])
        print(embeddings1, embeddings2)
        
        similarity = cosine_similarity(embeddings1, embeddings2)

        return similarity[0][0]  # Return the cosine similarity value


    # def __init__(self):
    #     model_name = './model-download/bert-base-uncased'
    #     cache_dir = './model-download'  # Specify a directory to store downloaded model files
    #     self.tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    #     self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2, cache_dir=cache_dir)
    #     self.optimizer = AdamW(self.model.parameters(), lr=1e-5)
    #     self.loss_fn = torch.nn.CrossEntropyLoss()

    def train(self, sentences1, sentences2, labels):
        inputs = self.tokenizer(sentences1, sentences2, padding=True, truncation=True, return_tensors='pt')
        labels_tensor = torch.tensor(labels)
        outputs = self.model(**inputs, labels=labels_tensor)
        loss = outputs.loss
        loss.backward()
        self.optimizer.step()

    # def predict(self, input_text1, input_text2):
    #     inputs = self.tokenizer(input_text1, input_text2, padding=True, truncation=True, return_tensors='pt')
    #     with torch.no_grad():
    #         logits = self.model(**inputs).logits
    #         prediction = torch.argmax(logits).item()
    #     result = "Entailment" if prediction == 1 else "Not Entailment"
    #     return result

    def predict1(self, sentence1, sentence2):
        # Tokenize and encode the sentences
        inputs1 = self.tokenizer(sentence1, return_tensors='pt', padding=True, truncation=True)
        inputs2 = self.tokenizer(sentence2, return_tensors='pt', padding=True, truncation=True)

        with torch.no_grad():
            outputs1 = self.model(**inputs1)
            outputs2 = self.model(**inputs2)
            print(outputs1)
            print(outputs2)

        # embeddings1 = outputs1.last_hidden_state[0, 0, :]  # [CLS] embedding for sentence 1
        # embeddings2 = outputs2.last_hidden_state[0, 0, :]  # [CLS] embedding for sentence 2

        # Calculate cosine similarity
        # similarity = cosine_similarity(embeddings1.unsqueeze(0), embeddings2.unsqueeze(0))
        # print(similarity[0][0])
        logits1 = outputs1.logits
        logits2 = outputs2.logits
        # Calculate cosine similarity based on logits
        similarity = cosine_similarity(logits1, logits2)

        return similarity[0][0]  # Return the cosine similarity value