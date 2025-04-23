import streamlit as st
import pandas as pd
import os

# Import custom modules
from preprocessing import preprocess_text, vectorize_data
from model_training import train_models
from model_evaluation import evaluate_models

# Page configuration
st.set_page_config(
    page_title="SpamShield - Email Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define logo and title
def display_header():
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("assets/shield_logo.svg", width=80)
    with col2:
        st.title("SpamShield: Email Spam Detection")
        st.subheader("Protect your inbox from unwanted emails")

# Function to load data and train models - we'll use this across pages
@st.cache_data
def load_and_train_models():
    try:
        # Load the spam.csv file
        with st.spinner("Loading and preprocessing data..."):
            df = pd.read_csv("attached_assets/spam.csv", encoding='latin-1')
            
            # Process the data format
            if 'v1' in df.columns and 'v2' in df.columns:
                df = df[['v1', 'v2']] 
                # Rename columns for consistency in our code
                df.columns = ['label', 'text']
            else:
                available_columns = ", ".join(df.columns)
                raise ValueError(f"Required columns 'v1' and 'v2' not found in dataset. Available columns: {available_columns}")
            
            # Preprocess text data
            df['processed_text'] = df['text'].apply(lambda x: preprocess_text(x, True, True, False))
            
            # Vectorize the data
            X, y, X_train, X_test, y_train, y_test, vectorizer = vectorize_data(
                df, 
                vectorizer_type="TF-IDF Vectorizer", 
                test_size=0.2, 
                random_state=42
            )
            
        # Train the models
        with st.spinner("Training machine learning models..."):
            models_to_train = ["Naive Bayes", "SVM", "Random Forest", "Logistic Regression"]
            models = train_models(X_train, y_train, models_to_train)
            
            # Evaluate models
            metrics = evaluate_models(models, X_test, y_test)
            
            # Calculate basic statistics for sidebar
            spam_count = df[df['label'] == 'spam'].shape[0]
            ham_count = df[df['label'] == 'ham'].shape[0]
            total_count = df.shape[0]
            
            # Store dataset for other pages
            dataset_info = {
                "df": df,
                "spam_count": spam_count,
                "ham_count": ham_count,
                "total_count": total_count
            }
            
            # Return everything needed for the application
            return models, vectorizer, X_test, y_test, metrics, dataset_info
    
    except Exception as e:
        st.error(f"Error during data loading and model training: {str(e)}")
        return None, None, None, None, None, None

# Initialize session state for sharing data between pages
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
if 'dataset_info' not in st.session_state:
    st.session_state.dataset_info = None
if 'email_text' not in st.session_state:
    st.session_state.email_text = ""

# Display page header
display_header()

# Load data and train models if not already done
if st.session_state.models is None:
    st.session_state.models, st.session_state.vectorizer, st.session_state.X_test, st.session_state.y_test, st.session_state.metrics, st.session_state.dataset_info = load_and_train_models()
    
    if st.session_state.models is not None:
        st.success("✅ Models trained successfully!")
    else:
        st.error("❌ Failed to train models. Please check the data file.")

# Sidebar with navigation and dataset stats
with st.sidebar:
    st.title("Navigation")
    st.write("📧 [Home - Email Detector](./)")
    st.write("📊 [Dataset Dashboard](./Dataset_Dashboard)")
    st.write("📈 [Model Performance](./Model_Performance)")
    
    if st.session_state.dataset_info:
        st.divider()
        st.subheader("Dataset Statistics")
        st.metric("Total Emails", st.session_state.dataset_info["total_count"])
        
        # Show spam/ham ratio using a simple text gauge
        spam_perc = st.session_state.dataset_info["spam_count"] / st.session_state.dataset_info["total_count"] * 100
        ham_perc = 100 - spam_perc
        st.write(f"**Spam ratio:** {spam_perc:.1f}%")
        st.progress(spam_perc/100)
        
        # If models are trained, show the best model by f1 score
        if st.session_state.metrics:
            st.divider()
            st.subheader("Model Performance")
            metrics_df = pd.DataFrame(st.session_state.metrics).T  # Transpose to get model names as index
            best_model = metrics_df["F1 Score"].idxmax()
            best_f1 = metrics_df.loc[best_model, "F1 Score"]
            st.write(f"**Best model:** {best_model}")
            st.write(f"**F1 Score:** {best_f1:.4f}")

# Main content - Email classification UI
st.write("## 📧 Email Spam Classification")
st.info("📌 Paste an email message below to check if it's spam or legitimate.")

# Main classification interface in a styled card
with st.container():
    st.markdown(
        """
        <style>
        .classification-container {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        </style>
        """, 
        unsafe_allow_html=True
    )
    
    # Example emails in expandable section
    with st.expander("Try with example emails"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Example Spam Email:**")
            example_spam = "URGENT: You have WON a $1,000 Walmart gift card. Go to http://claim-prize.com to claim now before it expires!"
            st.text_area("Spam Example", example_spam, height=100, key="spam_example", disabled=True)
            if st.button("Use Spam Example"):
                st.session_state.email_text = example_spam
                st.rerun()
        
        with col2:
            st.write("**Example Legitimate Email:**")
            example_ham = "Hi Sarah, just checking if we're still on for lunch tomorrow at 12:30pm? Let me know if you need to reschedule. Best, John"
            st.text_area("Legitimate Example", example_ham, height=100, key="ham_example", disabled=True)
            if st.button("Use Legitimate Example"):
                st.session_state.email_text = example_ham
                st.rerun()
    
    # Text input area
    user_input = st.text_area("Enter the email text to classify:", 
                            height=150, 
                            key="user_input",
                            value=st.session_state.email_text)
    
    col1, col2 = st.columns([3, 1])
    # Model selection
    if st.session_state.models is not None:
        model_options = list(st.session_state.models.keys())
        
        with col1:
            model_name = st.selectbox(
                "Select model for classification",
                model_options,
                index=2 if "Random Forest" in model_options else 0
            )
        
        with col2:
            st.write("")
            classify_button = st.button("Classify Email", type="primary", use_container_width=True)
        
        # Classification logic
        if classify_button and user_input:
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
                    st.divider()
                    st.subheader("Classification Result")
                    
                    # Custom HTML/CSS for better result display
                    if prediction == 1:  # Spam
                        st.markdown(f"""
                        <div style="background-color: #FFEBEE; border-left: 8px solid #F44336; padding: 15px; border-radius: 4px; margin: 10px 0;">
                            <h3 style="color: #D32F2F; margin:0;">📛 SPAM DETECTED!</h3>
                            <p style="color: #D32F2F; font-size: 18px; margin: 8px 0;">Confidence: {confidence:.2f}%</p>
                            <p>This email appears to be spam and may be attempting to deceive you.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:  # Ham
                        st.markdown(f"""
                        <div style="background-color: #E8F5E9; border-left: 8px solid #4CAF50; padding: 15px; border-radius: 4px; margin: 10px 0;">
                            <h3 style="color: #2E7D32; margin:0;">✅ LEGITIMATE EMAIL</h3>
                            <p style="color: #2E7D32; font-size: 18px; margin: 8px 0;">Confidence: {confidence:.2f}%</p>
                            <p>This email appears to be legitimate based on our analysis.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
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
                            
                            # Display top words in a more visual way
                            if word_importances:
                                for i, (word, importance) in enumerate(word_importances[:5], 1):
                                    st.markdown(f"""
                                    <div style="display: flex; align-items: center; margin-bottom: 5px;">
                                        <div style="width: 25px; text-align: center;">{i}</div>
                                        <div style="width: 80px; text-align: left; font-weight: bold;">{word}</div>
                                        <div style="flex-grow: 1; margin: 0 10px;">
                                            <div style="background-color: #E0E0E0; height: 10px; border-radius: 5px; width: 100%;">
                                                <div style="background-color: {'#F44336' if prediction == 1 else '#4CAF50'}; width: {min(importance*100, 100)}%; height: 10px; border-radius: 5px;"></div>
                                            </div>
                                        </div>
                                        <div style="width: 70px; text-align: right;">{importance:.4f}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.write("No specific words with high importance found in this message.")
                
                except Exception as e:
                    st.error(f"Error during classification: {str(e)}")
            
            # Save the text for convenience
            if user_input != st.session_state.email_text:
                st.session_state.email_text = user_input
        
        elif classify_button:
            st.warning("⚠️ Please enter some text to classify.")
    else:
        st.error("❌ Models are not available. Please check if the data loaded correctly in the sidebar.")

# Add footer
st.write("---")
st.caption("SpamShield - Email Spam Detection | Built with Streamlit and Machine Learning")