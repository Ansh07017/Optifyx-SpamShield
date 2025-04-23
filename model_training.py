from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import time

def train_models(X_train, y_train, models_to_train):
    """
    Train multiple machine learning models for spam detection.
    
    Args:
        X_train (scipy.sparse.csr.csr_matrix): The vectorized training features
        y_train (numpy.ndarray): The training labels
        models_to_train (list): List of model names to train
    
    Returns:
        dict: Dictionary of trained models
    """
    trained_models = {}
    
    for model_name in models_to_train:
        if model_name == "Naive Bayes":
            model = MultinomialNB()
        elif model_name == "SVM":
            model = SVC(kernel='linear', probability=True, random_state=42)
        elif model_name == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_name == "Logistic Regression":
            model = LogisticRegression(random_state=42, max_iter=1000)
        else:
            continue
        
        # Train the model and measure time
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Store the model and training time
        trained_models[model_name] = model
        
        # Print training time (for debugging)
        print(f"{model_name} trained in {training_time:.2f} seconds")
    
    return trained_models
