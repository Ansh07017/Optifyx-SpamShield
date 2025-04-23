from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import time
import sys

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
    
    print(f"Training {len(models_to_train)} models with {X_train.shape[0]} samples...")
    print(f"Features: {X_train.shape[1]}, X_train type: {type(X_train)}")
    print(f"Target: {len(y_train)}, target distribution: {sum(y_train)} spam, {len(y_train) - sum(y_train)} ham")
    
    for model_name in models_to_train:
        try:
            print(f"Starting to train {model_name}...")
            
            if model_name == "Naive Bayes":
                model = MultinomialNB()
            elif model_name == "SVM":
                model = SVC(kernel='linear', probability=True, random_state=42)
            elif model_name == "Random Forest":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif model_name == "Logistic Regression":
                model = LogisticRegression(random_state=42, max_iter=1000)
            else:
                print(f"Unknown model: {model_name}, skipping")
                continue
            
            # Train the model and measure time
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Store the model and training time
            trained_models[model_name] = model
            
            # Print training time (for debugging)
            print(f"{model_name} trained successfully in {training_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}", file=sys.stderr)
            # Continue with other models instead of failing completely
    
    print(f"Successfully trained {len(trained_models)} models")
    return trained_models
