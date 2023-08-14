from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import secrets

db = SQLAlchemy()
migrate = Migrate()

def create_app():
    app = Flask(__name__)

    secret_key = secrets.token_hex(16)  # Generate a 32-character secret key
    # Set the secret key for CSRF protection
    app.config['SECRET_KEY'] = secret_key

    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'  # SQLite database URI
    db.init_app(app)  # Bind SQLAlchemy to the app
    migrate.init_app(app, db)  # Initialize Flask-Migrate

    from .routes import bp
    app.register_blueprint(bp)

    return app

