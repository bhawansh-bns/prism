from flask import Flask
import secrets

def create_app():
    app = Flask(__name__)

    secret_key = secrets.token_hex(16)  # Generate a 32-character secret key
    # Set the secret key for CSRF protection
    app.config['SECRET_KEY'] = secret_key

    from .routes import bp
    app.register_blueprint(bp)

    return app

