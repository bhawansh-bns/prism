from flask import Blueprint, render_template, request
from .bert_model import BERTModel
from .forms import PredictionForm
from .models import SentencePair
from app import db  # Import db from app module

bp = Blueprint('main', __name__)

model = BERTModel()  # Initialize the BERT model

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
    from .train_model import train_cosine_similarity_model

    # Call the training function and pass your model instance
    train_cosine_similarity_model(model)
    return "Training completed and model saved."


