# HOW TO START THE APPLICATION
git clone https://github.com/bhawansh-bns/prism.git<br>
cd <repository_directory><br>
python -m venv venv<br>
venv\Scripts\activate<br>
pip install -r requirements.txt<br>
flask db init<br>
flask db migrate -m "Initial migration"<br>
flask db upgrade<br>
flask run<br>