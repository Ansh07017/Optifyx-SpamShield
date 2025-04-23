import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import time
import base64
from io import StringIO
import os

# Import custom modules
from preprocessing import preprocess_text, vectorize_data
from model_training import train_models
from model_evaluation import evaluate_models, plot_confusion_matrices, plot_roc_curves

# Page configuration
st.set_page_config(
    page_title="SpamShield - Email Spam Detection",
    page_icon="🛡️",
    layout="wide",
)

# Define logo and title
def display_header():
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("assets/shield_logo.svg", width=80)
    with col2:
        st.title("SpamShield: Email Spam Detection")
        st.subheader("Protect your inbox from unwanted emails")

# Function to load data and train models
@st.cache_data
def load_and_train_models():
    try:
        # Load the spam.csv file
        st.info("Loading spam dataset...")
        df = pd.read_csv("attached_assets/spam.csv", encoding='latin-1')
        
        # Process the data format
        st.info("Processing dataset...")
        # Keep only the first two columns (v1, v2) from the CSV file
        # We can see from the screenshot that v1 contains 'ham'/'spam' labels and v2 contains the email text
        if 'v1' in df.columns and 'v2' in df.columns:
            df = df[['v1', 'v2']] 
            # Rename columns for consistency in our code
            df.columns = ['label', 'text']
            st.success(f"Successfully loaded data with {df.shape[0]} rows")
        else:
            available_columns = ", ".join(df.columns)
            raise ValueError(f"Required columns 'v1' and 'v2' not found in dataset. Available columns: {available_columns}")
        
        # Display the first few rows to verify the data format
        st.write("First 5 rows of the dataset:")
        st.dataframe(df.head())
        
        # Display dataset info
        st.write("### Dataset Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Emails", df.shape[0])
        with col2:
            spam_count = df[df['label'] == 'spam'].shape[0]
            st.metric("Spam Emails", spam_count)
        with col3:
            ham_count = df[df['label'] == 'ham'].shape[0]
            st.metric("Ham Emails", ham_count)
        
        # Display sample data
        st.write("### Dataset Preview")
        st.dataframe(df.head())
        
        # Data visualization
        st.write("### Data Distribution")
        fig = px.pie(values=[spam_count, ham_count], 
                    names=['Spam', 'Ham'], 
                    title="Distribution of Spam vs. Ham",
                    color_discrete_sequence=['#ff6b6b', '#4ecdc4'])
        st.plotly_chart(fig)
        
        # Email length analysis
        st.write("### Email Length Analysis")
        df['text_length'] = df['text'].apply(len)
        fig = px.histogram(df, x='text_length', color='label',
                         nbins=50, opacity=0.7,
                         color_discrete_map={'spam': '#ff6b6b', 'ham': '#4ecdc4'},
                         title="Distribution of Email Lengths by Class")
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig)
        
        # Preprocess the data
        st.write("### Preprocessing and Training Models...")
        st.info("Preprocessing text data...")
        try:
            df['processed_text'] = df['text'].apply(lambda x: preprocess_text(x, True, True, False))
            
            # Vectorize the data
            st.info("Vectorizing data...")
            X, y, X_train, X_test, y_train, y_test, vectorizer = vectorize_data(
                df, 
                vectorizer_type="TF-IDF Vectorizer", 
                test_size=0.2, 
                random_state=42
            )
            
            # Train the models
            st.info("Training machine learning models...")
            models_to_train = ["Naive Bayes", "SVM", "Random Forest", "Logistic Regression"]
            models = train_models(X_train, y_train, models_to_train)
        except Exception as e:
            st.error(f"Error during preprocessing or training: {str(e)}")
            raise e
        
        # Evaluate the models
        metrics = evaluate_models(models, X_test, y_test)
        
        # Display metrics in a table
        st.write("### Model Performance Metrics")
        metrics_df = pd.DataFrame(metrics)
        st.dataframe(metrics_df)
        
        # Create bar chart for model comparison
        fig = px.bar(
            metrics_df, 
            barmode='group',
            color_discrete_sequence=px.colors.qualitative.G10,
            title="Performance Metrics Comparison"
        )
        st.plotly_chart(fig)
        
        # Display confusion matrices
        st.write("### Confusion Matrices")
        fig = plot_confusion_matrices(models, X_test, y_test)
        st.pyplot(fig)
        
        # Display ROC curves
        st.write("### ROC Curves")
        fig = plot_roc_curves(models, X_test, y_test)
        st.pyplot(fig)
        
        # Return results
        return models, vectorizer, X_test, y_test, metrics
    
    except Exception as e:
        st.error(f"Error during data loading and model training: {str(e)}")
        return None, None, None, None, None

# Initialize session state variables
if 'models' not in st.session_state:
    st.session_state.models = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None

# Display header
display_header()

# Load data and train models if not already done
if st.session_state.models is None:
    with st.spinner("Loading data and training models... This may take a minute or two."):
        st.session_state.models, st.session_state.vectorizer, st.session_state.X_test, st.session_state.y_test, st.session_state.metrics = load_and_train_models()
    
    if st.session_state.models is not None:
        st.success("✅ Models trained successfully!")
    else:
        st.error("❌ Failed to train models. Please check the data file.")

# Add a separator
st.write("---")

# Email classification UI
st.write("## 📧 Email Spam Classification")
st.write("Paste an email message below to check if it's spam or legitimate.")

# Example emails in expandable section
with st.expander("Try with example emails"):
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Example Spam Email:**")
        example_spam = "URGENT: You have WON a $1,000 Walmart gift card. Go to http://claim-prize.com to claim now before it expires!"
        st.text_area("Spam Example", example_spam, height=100, key="spam_example", disabled=True)
        if st.button("Use Spam Example"):
            st.session_state.email_text = example_spam
    
    with col2:
        st.write("**Example Legitimate Email:**")
        example_ham = "Hi Sarah, just checking if we're still on for lunch tomorrow at 12:30pm? Let me know if you need to reschedule. Best, John"
        st.text_area("Legitimate Example", example_ham, height=100, key="ham_example", disabled=True)
        if st.button("Use Legitimate Example"):
            st.session_state.email_text = example_ham

# Initialize email text session state if not exists
if 'email_text' not in st.session_state:
    st.session_state.email_text = ""

# Text input for classification
user_input = st.text_area("Enter the email text to classify:", 
                          height=150, 
                          key="user_input",
                          value=st.session_state.email_text)

# Model selection for classification
if st.session_state.models is not None:
    model_options = list(st.session_state.models.keys())
    col1, col2 = st.columns([3, 1])
    
    with col1:
        model_name = st.selectbox(
            "Select model for classification",
            model_options,
            index=2 if "Random Forest" in model_options else 0
        )
    
    with col2:
        st.write("")
        st.write("")
        classify_button = st.button("Classify Email", type="primary")
    
    if classify_button:
        if user_input:
            with st.spinner("Analyzing email..."):
                try:
                    # Preprocess the input text
                    processed_text = preprocess_text(user_input)
                    
                    # Vectorize the text
                    vectorized_text = st.session_state.vectorizer.transform([processed_text])
                    
                    # Get the selected model
                    model = st.session_state.models[model_name]
                    
                    # Make prediction
                    prediction = model.predict(vectorized_text)[0]
                    
                    # Get prediction probability
                    try:
                        prob = model.predict_proba(vectorized_text)[0]
                        confidence = max(prob) * 100
                    except:
                        confidence = 99  # Default if not available
                    
                    # Display result with animation
                    st.subheader("Classification Result")
                    
                    if prediction == 1:  # Spam
                        with st.container():
                            st.error(f"📛 **SPAM DETECTED!** This email is classified as spam with {confidence:.2f}% confidence.")
                            st.warning("⚠️ This email appears to be spam and may be attempting to deceive you.")
                    else:  # Ham
                        with st.container():
                            st.success(f"✅ **LEGITIMATE EMAIL** This email is classified as legitimate with {confidence:.2f}% confidence.")
                            st.info("💡 This email appears to be legitimate based on our analysis.")
                    
                    # Display explanation if using Random Forest
                    if model_name == "Random Forest":
                        with st.expander("Why was this classified this way?"):
                            st.write("The Random Forest model identified these key elements in making this classification:")
                            
                            # Get feature names and importances
                            feature_names = st.session_state.vectorizer.get_feature_names_out()
                            importances = model.feature_importances_
                            
                            # Get the words in the email
                            words = processed_text.split()
                            
                            # Find feature importance for words in the email
                            word_importances = []
                            for word in set(words):
                                if word in feature_names:
                                    idx = list(feature_names).index(word)
                                    word_importances.append((word, importances[idx]))
                            
                            # Sort by importance
                            word_importances.sort(key=lambda x: x[1], reverse=True)
                            
                            # Display top 5 important words
                            if word_importances:
                                for word, importance in word_importances[:5]:
                                    st.write(f"- '{word}' (importance: {importance:.4f})")
                            else:
                                st.write("No specific words with high importance found in this message.")
                
                except Exception as e:
                    st.error(f"Error during classification: {str(e)}")
        else:
            st.warning("Please enter some text to classify.")
else:
    st.error("Models are not available. Please check if the data loaded correctly.")

# Add footer
st.write("---")
st.caption("SpamShield - Email Spam Detection | Built with Streamlit and Machine Learning")