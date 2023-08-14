from app import db

class SentencePair(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sentence1 = db.Column(db.String(255), nullable=False)
    sentence2 = db.Column(db.String(255), nullable=False)
    cosine_similarity = db.Column(db.Float, nullable=False)