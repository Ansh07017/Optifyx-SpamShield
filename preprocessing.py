import nltk
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('omw-1.4', quiet=True)

def preprocess_text(text, remove_stopwords=True, stemming=True, lemmatization=False):
    """
    Preprocess the text by cleaning, tokenizing, and applying various NLP techniques.
    
    Args:
        text (str): The input text to preprocess
        remove_stopwords (bool): Whether to remove stopwords
        stemming (bool): Whether to apply stemming
        lemmatization (bool): Whether to apply lemmatization
    
    Returns:
        str: The preprocessed text
    """
    # Convert to string if not already
    if not isinstance(text, str):
        text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    
    # Remove stopwords if specified
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
    
    # Apply stemming if specified
    if stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
    
    # Apply lemmatization if specified
    if lemmatization:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens back into a single string
    processed_text = ' '.join(tokens)
    
    return processed_text

def vectorize_data(df, vectorizer_type='TF-IDF Vectorizer', test_size=0.2, random_state=42):
    """
    Vectorize the text data for machine learning models.
    
    Args:
        df (pandas.DataFrame): The dataframe containing the text data
        vectorizer_type (str): The type of vectorizer to use
        test_size (float): The proportion of the dataset to include in the test split
        random_state (int): Random state for reproducibility
    
    Returns:
        tuple: X, y, X_train, X_test, y_train, y_test, vectorizer
    """
    # Get features and target
    X = df['processed_text'] if 'processed_text' in df.columns else df['text'].apply(preprocess_text)
    y = df['label']
    
    # Label encoding: convert 'spam'/'ham' to 1/0
    y = np.where(y == 'spam', 1, 0)
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Initialize the vectorizer based on the selected type
    if vectorizer_type == 'CountVectorizer':
        vectorizer = CountVectorizer(max_features=5000)
    else:  # TF-IDF Vectorizer
        vectorizer = TfidfVectorizer(max_features=5000)
    
    # Fit and transform the training data
    X_train_vec = vectorizer.fit_transform(X_train)
    
    # Transform the test data
    X_test_vec = vectorizer.transform(X_test)
    
    return X, y, X_train_vec, X_test_vec, y_train, y_test, vectorizer
