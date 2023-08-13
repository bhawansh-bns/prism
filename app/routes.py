from flask import Blueprint, render_template, request
from .bert_model import BERTModel
from .forms import PredictionForm

bp = Blueprint('main', __name__)

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
        return render_template('index.html', form=form, similarity=similarity)
    return render_template('index.html', form=form)

model = BERTModel()  # Initialize the BERT model


