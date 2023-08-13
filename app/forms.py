from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

class PredictionForm(FlaskForm):
    sentence1 = StringField('Sentence 1', validators=[DataRequired()])
    sentence2 = StringField('Sentence 2', validators=[DataRequired()])
    submit = SubmitField('Predict')
