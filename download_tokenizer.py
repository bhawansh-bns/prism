from transformers import BertTokenizer

model_name = 'bert-base-uncased'
cache_dir = './model-download'
tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

print("Tokenizer loaded successfully.")
