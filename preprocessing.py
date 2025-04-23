import re
import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

# Define a list of common English stopwords directly in the code to avoid NLTK dependency
ENGLISH_STOPWORDS = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
                      "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 
                      'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 
                      'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 
                      'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 
                      'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 
                      'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 
                      'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 
                      'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 
                      'with', 'about', 'against', 'between', 'into', 'through', 'during', 
                      'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 
                      'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 
                      'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 
                      'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 
                      'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 
                      'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 
                      "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 
                      've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', 
                      "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 
                      'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 
                      'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', 
                      "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 
                      'wouldn', "wouldn't"])

# Create a simple Porter Stemmer implementation
class SimpleStemmer:
    """A simple implementation of the Porter stemming algorithm."""
    
    def stem(self, word):
        """Return the stem of a word."""
        # This is a very simplified implementation
        # Just handle some common endings
        if len(word) > 3:
            if word.endswith('ing'):
                return word[:-3]
            elif word.endswith('ed'):
                return word[:-2]
            elif word.endswith('ly'):
                return word[:-2]
            elif word.endswith('s'):
                return word[:-1]
        return word

# Create a simple stemmer
stemmer = SimpleStemmer()

try:
    # Try to import NLTK for better NLP functionality
    import nltk
    from nltk.stem import PorterStemmer
    
    # Create NLTK data directory if it doesn't exist
    nltk_data_dir = os.path.join(os.path.expanduser('~'), 'nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    
    # Try to download stopwords resource
    try:
        nltk.download('stopwords', quiet=True, download_dir=nltk_data_dir)
        from nltk.corpus import stopwords
        ENGLISH_STOPWORDS = set(stopwords.words('english'))
        print("Using NLTK stopwords")
    except Exception as e:
        print(f"Using built-in stopwords. NLTK stopwords unavailable: {str(e)}")
    
    # Try to get a better stemmer
    try:
        stemmer = PorterStemmer()
        print("Using NLTK PorterStemmer")
    except Exception as e:
        print(f"Using simplified stemmer. NLTK stemmer unavailable: {str(e)}")
        
    print("NLTK imported successfully")
except ImportError:
    print("NLTK import failed. Using simplified NLP features.")

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
    
    # Simple tokenization by splitting on whitespace to avoid NLTK word_tokenize issues
    tokens = text.split()
    
    # Remove stopwords if specified
    if remove_stopwords:
        # Use our predefined stopwords list
        tokens = [token for token in tokens if token not in ENGLISH_STOPWORDS]
    
    # Apply stemming if specified
    if stemming:
        # Use the stemmer that was defined earlier (either NLTK or our simple one)
        tokens = [stemmer.stem(token) for token in tokens]
    
    # Apply lemmatization if specified
    # We'll skip lemmatization in our simplified version since it requires NLTK WordNet
    # and just use stemming instead
    if lemmatization:
        print("Lemmatization requires NLTK WordNet - skipping")
        # If lemmatization is requested but we don't have it, apply stemming instead
        if not stemming:  # Only if stemming wasn't already applied
            tokens = [stemmer.stem(token) for token in tokens]
    
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
