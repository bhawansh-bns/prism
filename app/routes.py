from flask import Blueprint, render_template, request
# from .bert_model import BERTModel
from .forms import PredictionForm
from .models import SentencePair
from app import db  # Import db from app module
from .sentence_transformers import SentenceTransformersModel

bp = Blueprint('main', __name__)

model = SentenceTransformersModel()  # Initialize the BERT model

@bp.route('/')
def index():
    form = PredictionForm()
    return render_template('index.html', form=form)

@bp.route('/predict', methods=['POST'])
def predict():
    form = PredictionForm()
    similarity = None  # Initialize similarity as None

    if form.validate_on_submit():
        input_text1 = form.sentence1.data
        input_text2 = form.sentence2.data
        similarity = model.predict(input_text1, input_text2)
        
        # Check if a similar entry already exists in the database
        existing_entry = SentencePair.query.filter_by(sentence1=input_text1, sentence2=input_text2).first()
        if existing_entry:
            similarity = existing_entry.cosine_similarity
        else:
            similarity = model.predict(input_text1, input_text2)
            # Store data in the database
            sentence_pair = SentencePair(sentence1=input_text1, sentence2=input_text2, cosine_similarity=similarity)
            db.session.add(sentence_pair)
            db.session.commit()

        return render_template('index.html', form=form, similarity=similarity)
    return render_template('index.html', form=form)

@bp.route('/history')
def history():
    sentence_pairs = SentencePair.query.all()  # Retrieve all entries from the database
    return render_template('history.html', sentence_pairs=sentence_pairs)

@bp.route('/train', methods=['GET'])
def train():
    sentence_pairs_from_db = SentencePair.query.all()
    # sentence_pairs = [
    #     SentencePair(sentence1=pair.sentence1, sentence2=pair.sentence2, cosine_similarity=pair.cosine_similarity)
    #     for pair in sentence_pairs_from_db
    #     ]
    # print(sentence_pairs)
    # sentence_pairs = [
    #         ('Set the temperature to 72 degrees', 'Adjust thermostat to 72 degrees', 0.8340750932693481),
    #         ('Turn on the lamp in the living room', 'Switch on the living room lamp', 0.8756259679794312),
    #         ('Lock the back door', 'Secure the rear entrance', 0.8282058835029602),
    # ]
    model.train_model(sentence_pairs_from_db)
    return "Training completed and model saved."


