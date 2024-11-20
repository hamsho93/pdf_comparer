import os

# Flask Configurations
SECRET_KEY = os.environ.get('SECRET_KEY', 'default_secret_key')  # Use an environment variable for security
DEBUG = True
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')  # Set your upload folder path
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # Max file size: 16MB
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
