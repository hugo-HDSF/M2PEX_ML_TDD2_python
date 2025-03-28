import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import joblib
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from app import logger
from app.database.Tweet.repository import TweetRepository
from app.scripts.pdf_generator import generate_evaluation_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk import word_tokenize
from datetime import datetime

# Download required NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab')
nltk.download('wordnet', quiet=True)

# Define paths for models and metrics
MODEL_SIMPLE_PATH = 'app/data/ml_model.pkl'
MODEL_ADVANCED_PATH = 'app/data/nlp_model.pkl'
VECTORIZER_SIMPLE_PATH = 'app/data/tfidf_vectorizer_ml.pkl'
VECTORIZER_ADVANCED_PATH = 'app/data/tfidf_vectorizer_nlp.pkl'
METRICS_PATH = 'app/static/metrics.json'
CONFUSION_MATRIX_SIMPLE_PATH = 'app/static/confusion_matrix_ml.png'
CONFUSION_MATRIX_ADVANCED_PATH = 'app/static/confusion_matrix_nlp.png'
COMPARISON_PATH = 'app/static/model_comparison.png'


def remove_stopwords(text, language='french'):
    """
    Remove stopwords from text using NLTK

    Args:
        text (str): Input text
        language (str): Language for stopwords (default: french)

    Returns:
        str: Text with stopwords removed
    """
    # Handle None or non-string values
    if not isinstance(text, str):
        return ""

    stop_words = set(stopwords.words(language))
    word_tokens = word_tokenize(text.lower(), language=language)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_text)


def clean_string(text):
    """
    Clean the input string by removing special characters and extra spaces.

    Args:
        text (str): The input string to clean.
    Returns:
        str: The cleaned string.
    """
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    # Then, replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)

    # Optional: strip leading and trailing spaces
    text = text.strip()

    return text


def lemmatize_and_stem(text, language='french'):
    """
    Apply lemmatization and stemming to the input text

    Args:
        text (str): The input string to process
        language (str): Language for stemming (default: french)

    Returns:
        str: Processed text
    """
    if not isinstance(text, str):
        return ""

    # Initialize lemmatizer and stemmer
    lemmatizer = WordNetLemmatizer()
    stemmer = SnowballStemmer(language)

    # Tokenize
    word_tokens = word_tokenize(text.lower(), language=language)

    # Lemmatize and stem each token
    processed_tokens = [stemmer.stem(lemmatizer.lemmatize(word)) for word in word_tokens]

    return ' '.join(processed_tokens)


def train_model():
    """
    Train both sentiment analysis models using data from the database:
    1. Simple model: Single logistic regression with basic preprocessing
    2. Advanced model: Single logistic regression with lemmatization and stemming
    """
    logger.info("Starting model training for both models")

    try:
        # Fetch data from database
        tweets = TweetRepository.get_all()

        if len(tweets) < 10:
            logger.warning("Not enough data to train models")
            return False

        df = pd.DataFrame([(t.text, t.positive, t.negative) for t in tweets],
                          columns=['text', 'positive', 'negative'])

        # Create a single target variable from positive and negative labels
        # 1 = positive, 0 = neutral, -1 = negative
        df['sentiment'] = 0
        df.loc[df['positive'] == 1, 'sentiment'] = 1
        df.loc[df['negative'] == 1, 'sentiment'] = -1

        # ---- SIMPLE MODEL TRAINING ----
        logger.info("Training simple model with basic preprocessing")

        # Basic preprocessing
        df['processed_text'] = df['text'].apply(lambda x: remove_stopwords(clean_string(x), 'french'))

        # Prepare data
        X = df['processed_text']
        y = df['sentiment']

        # Create TF-IDF vectorizer for simple model
        vectorizer_simple = TfidfVectorizer(max_features=1000)
        X_tfidf_simple = vectorizer_simple.fit_transform(X)

        # Split data 80/20
        X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(
            X_tfidf_simple, y, test_size=0.2, random_state=42)

        # Train simple model
        model_simple = LogisticRegression(max_iter=1000, multi_class='ovr')
        model_simple.fit(X_train_simple, y_train_simple)

        # Evaluate simple model
        y_pred_simple = model_simple.predict(X_test_simple)

        # Calculate metrics for simple model
        accuracy_simple = accuracy_score(y_test_simple, y_pred_simple)
        precision_simple, recall_simple, f1_simple, _ = precision_recall_fscore_support(
            y_test_simple, y_pred_simple, average='weighted')

        # Generate confusion matrix for simple model
        plt.figure(figsize=(8, 6))
        cm_simple = confusion_matrix(y_test_simple, y_pred_simple)
        sns.heatmap(cm_simple, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Neutral', 'Positive'],
                    yticklabels=['Negative', 'Neutral', 'Positive'])
        plt.title('Confusion Matrix - Simple Model')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(CONFUSION_MATRIX_SIMPLE_PATH)
        plt.close()

        # ---- ADVANCED MODEL TRAINING ----
        logger.info("Training advanced model with lemmatization and stemming")

        # Advanced preprocessing with lemmatization and stemming
        df['advanced_processed_text'] = df['processed_text'].apply(
            lambda x: lemmatize_and_stem(x, 'french'))

        # Create TF-IDF vectorizer for advanced model
        vectorizer_advanced = TfidfVectorizer(max_features=1000)
        X_tfidf_advanced = vectorizer_advanced.fit_transform(df['advanced_processed_text'])

        # Split data 80/20
        X_train_advanced, X_test_advanced, y_train_advanced, y_test_advanced = train_test_split(
            X_tfidf_advanced, y, test_size=0.2, random_state=42)

        # Train advanced model
        model_advanced = LogisticRegression(max_iter=1000, multi_class='ovr')
        model_advanced.fit(X_train_advanced, y_train_advanced)

        # Evaluate advanced model
        y_pred_advanced = model_advanced.predict(X_test_advanced)

        # Calculate metrics for advanced model
        accuracy_advanced = accuracy_score(y_test_advanced, y_pred_advanced)
        precision_advanced, recall_advanced, f1_advanced, _ = precision_recall_fscore_support(
            y_test_advanced, y_pred_advanced, average='weighted')

        # Generate confusion matrix for advanced model
        plt.figure(figsize=(8, 6))
        cm_advanced = confusion_matrix(y_test_advanced, y_pred_advanced)
        sns.heatmap(cm_advanced, annot=True, fmt='d', cmap='Greens',
                    xticklabels=['Negative', 'Neutral', 'Positive'],
                    yticklabels=['Negative', 'Neutral', 'Positive'])
        plt.title('Confusion Matrix - Advanced Model')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(CONFUSION_MATRIX_ADVANCED_PATH)
        plt.close()

        # --- MODEL COMPARISON VISUALIZATION ---
        plt.figure(figsize=(10, 6))
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        simple_scores = [accuracy_simple, precision_simple, recall_simple, f1_simple]
        advanced_scores = [accuracy_advanced, precision_advanced, recall_advanced, f1_advanced]

        x = range(len(metrics))
        width = 0.35

        plt.bar([i - width / 2 for i in x], simple_scores, width, label='Simple Model')
        plt.bar([i + width / 2 for i in x], advanced_scores, width, label='Advanced Model')

        plt.ylim(0, 1.0)
        plt.title('Model Comparison')
        plt.ylabel('Score')
        plt.xticks(x, metrics)
        plt.legend()
        plt.tight_layout()
        plt.savefig(COMPARISON_PATH)
        plt.close()

        # Save metrics
        metrics = {
            'simple_model': {
                'accuracy': float(accuracy_simple),
                'precision': float(precision_simple),
                'recall': float(recall_simple),
                'f1': float(f1_simple),
            },
            'advanced_model': {
                'accuracy': float(accuracy_advanced),
                'precision': float(precision_advanced),
                'recall': float(recall_advanced),
                'f1': float(f1_advanced),
            },
            'last_trained': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(METRICS_PATH, 'w') as f:
            json.dump(metrics, f)

        # Save models and vectorizers
        joblib.dump(model_simple, MODEL_SIMPLE_PATH)
        joblib.dump(vectorizer_simple, VECTORIZER_SIMPLE_PATH)
        joblib.dump(model_advanced, MODEL_ADVANCED_PATH)
        joblib.dump(vectorizer_advanced, VECTORIZER_ADVANCED_PATH)

        # Generate comparison report
        generate_evaluation_report(
            y_test_simple, y_pred_simple,
            y_test_advanced, y_pred_advanced,
            metrics
        )

        logger.info("Model training completed successfully for both models")
        return True
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        return False


def predict_sentiment(text, model_type='simple'):
    """
    Predict the sentiment of the given text using the specified model.

    Args:
        text (str): The text to analyze
        model_type (str): Which model to use ('simple' or 'advanced')

    Returns:
        float: A sentiment score between -1 (very negative) and 1 (very positive)
    """
    if model_type not in ['simple', 'advanced']:
        logger.warning(f"Invalid model type: {model_type}. Using simple model as fallback.")
        model_type = 'simple'

    model_path = MODEL_SIMPLE_PATH if model_type == 'simple' else MODEL_ADVANCED_PATH
    vectorizer_path = VECTORIZER_SIMPLE_PATH if model_type == 'simple' else VECTORIZER_ADVANCED_PATH

    # Check if model exists, if not train it
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        if not train_model():
            # If training failed, return neutral sentiment
            return 0

    # Load model
    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)

        # Preprocess text
        processed_text = remove_stopwords(clean_string(text), 'french')

        # Apply additional preprocessing if using advanced model
        if model_type == 'advanced':
            processed_text = lemmatize_and_stem(processed_text, 'french')

        # Vectorize text
        X = vectorizer.transform([processed_text])

        # Predict sentiment class (-1, 0, 1)
        sentiment_class = model.predict(X)[0]

        # Get probability scores
        probs = model.predict_proba(X)[0]

        # Log prediction details
        logger.info(f"Model: {model_type}, Raw prediction: {sentiment_class}")
        logger.info(f"Sentiment analysis: '{text}' - Result: {sentiment_class}")

        return float(sentiment_class)  # Convert to float to ensure JSON serialization
    except Exception as e:
        logger.error(f"Error predicting sentiment with {model_type} model: {str(e)}")
        return 0  # Return neutral sentiment if there's an error
