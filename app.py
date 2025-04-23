import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import time
import base64
from io import StringIO

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

# Initialize session state variables
if 'models' not in st.session_state:
    st.session_state.models = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None

# Display header
display_header()

# Create sidebar
st.sidebar.markdown("## Navigation")
page = st.sidebar.radio("Select a page:", ["Data Upload & Analysis", "Model Training & Evaluation", "Spam Classification"])

# Main content based on selected page
if page == "Data Upload & Analysis":
    st.header("📊 Data Upload & Analysis")
    st.write("Upload your dataset to analyze and prepare for training spam detection models.")
    
    # Upload file
    uploaded_file = st.file_uploader("Upload your CSV file containing email data", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Read CSV and display sample
            df = pd.read_csv(uploaded_file)
            
            # Check if the required columns are present
            missing_cols = []
            if 'text' not in df.columns and 'message' not in df.columns:
                missing_cols.append('text/message')
            if 'label' not in df.columns and 'class' not in df.columns and 'category' not in df.columns:
                missing_cols.append('label/class/category')
            
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}. Please ensure your dataset has columns for the email text and spam/ham label.")
            else:
                # Standardize column names
                if 'text' not in df.columns and 'message' in df.columns:
                    df['text'] = df['message']
                if 'label' not in df.columns:
                    if 'class' in df.columns:
                        df['label'] = df['class']
                    elif 'category' in df.columns:
                        df['label'] = df['category']
                
                # Ensure only required columns are kept
                df = df[['text', 'label']]
                
                # Display dataset info
                st.subheader("Dataset Preview")
                st.dataframe(df.head())
                
                st.subheader("Dataset Information")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Emails", df.shape[0])
                with col2:
                    spam_count = df[df['label'] == 'spam'].shape[0]
                    st.metric("Spam Emails", spam_count)
                with col3:
                    ham_count = df[df['label'] == 'ham'].shape[0]
                    st.metric("Ham Emails", ham_count)
                
                # Data visualization
                st.subheader("Data Distribution")
                fig = px.pie(values=[spam_count, ham_count], 
                             names=['Spam', 'Ham'], 
                             title="Distribution of Spam vs. Ham",
                             color_discrete_sequence=['#ff6b6b', '#4ecdc4'])
                st.plotly_chart(fig)
                
                # Text length analysis
                st.subheader("Text Length Analysis")
                df['text_length'] = df['text'].apply(len)
                
                fig = px.histogram(df, x='text_length', color='label',
                                 nbins=50, opacity=0.7,
                                 color_discrete_map={'spam': '#ff6b6b', 'ham': '#4ecdc4'},
                                 title="Distribution of Email Lengths by Class")
                fig.update_layout(bargap=0.1)
                st.plotly_chart(fig)
                
                # Save the dataframe to session state
                st.session_state.df = df
                st.session_state.data_loaded = True
                
                st.success("Dataset successfully loaded! You can now proceed to Model Training & Evaluation.")
                
        except Exception as e:
            st.error(f"Error: {e}")
            st.error("Please make sure the file is a valid CSV with proper formatting.")
    
    else:
        st.info("Please upload a CSV file to continue.")
        
        # Option for using sample dataset
        if st.button("Use Sample Dataset"):
            # Create a sample dataset
            data = {
                'text': [
                    "Congratulations! You've won a free iPhone. Click here to claim now!",
                    "Meeting scheduled for tomorrow at 10 AM. Please bring your reports.",
                    "URGENT: Your account has been compromised. Verify your details now!",
                    "Hi Mom, how are you doing? I'll call you this evening.",
                    "FREE VIAGRA! Special discount for new customers. Order now!",
                    "Your Amazon order #12345 has been shipped and will arrive tomorrow.",
                    "You have won $1,000,000 in the international lottery. Send details to claim.",
                    "Reminder: Doctor's appointment on Friday at 2:30 PM.",
                    "SPECIAL OFFER: 80% OFF on all products! Limited time only!",
                    "The project deadline has been extended to next Monday."
                ],
                'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham']
            }
            
            sample_df = pd.DataFrame(data)
            st.session_state.df = sample_df
            st.session_state.data_loaded = True
            
            # Display sample dataset
            st.subheader("Sample Dataset Preview")
            st.dataframe(sample_df)
            
            st.subheader("Dataset Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Emails", sample_df.shape[0])
            with col2:
                spam_count = sample_df[sample_df['label'] == 'spam'].shape[0]
                st.metric("Spam Emails", spam_count)
            with col3:
                ham_count = sample_df[sample_df['label'] == 'ham'].shape[0]
                st.metric("Ham Emails", ham_count)
            
            st.success("Sample dataset loaded! You can now proceed to Model Training & Evaluation.")

elif page == "Model Training & Evaluation":
    st.header("🔍 Model Training & Evaluation")
    
    if not st.session_state.data_loaded:
        st.warning("Please upload a dataset first on the Data Upload & Analysis page.")
    else:
        st.write("Train and evaluate multiple machine learning models for spam detection.")
        
        # Model configuration
        st.subheader("Model Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Test Set Size (%)", 10, 40, 20) / 100
            random_state = st.number_input("Random State", 0, 100, 42)
            
        with col2:
            models_to_train = st.multiselect(
                "Select Models to Train",
                ["Naive Bayes", "SVM", "Random Forest", "Logistic Regression"],
                ["Naive Bayes", "SVM", "Random Forest"]
            )
        
        # Text preprocessing options
        st.subheader("Text Preprocessing Options")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            remove_stopwords = st.checkbox("Remove Stopwords", True)
        with col2:
            stemming = st.checkbox("Apply Stemming", True)
        with col3:
            lemmatization = st.checkbox("Apply Lemmatization", False)
        
        # Feature extraction options
        st.subheader("Feature Extraction")
        vectorizer_type = st.selectbox(
            "Vectorizer Type",
            ["CountVectorizer", "TF-IDF Vectorizer"],
            index=1
        )
        
        # Train models button
        if st.button("Train Models"):
            with st.spinner("Training models... This may take a while."):
                try:
                    # Preprocess the data
                    df = st.session_state.df
                    df['processed_text'] = df['text'].apply(lambda x: preprocess_text(x, remove_stopwords, stemming, lemmatization))
                    
                    # Vectorize the data
                    X, y, X_train, X_test, y_train, y_test, vectorizer = vectorize_data(
                        df, 
                        vectorizer_type=vectorizer_type, 
                        test_size=test_size, 
                        random_state=random_state
                    )
                    
                    # Train the selected models
                    models = train_models(X_train, y_train, models_to_train)
                    
                    # Evaluate the models
                    metrics = evaluate_models(models, X_test, y_test)
                    
                    # Store the results in session state
                    st.session_state.models = models
                    st.session_state.vectorizer = vectorizer
                    st.session_state.metrics = metrics
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    
                    st.success("Models trained successfully!")
                    
                except Exception as e:
                    st.error(f"Error during model training: {str(e)}")
        
        # Display model evaluation if models have been trained
        if st.session_state.models is not None and st.session_state.metrics is not None:
            st.header("Model Evaluation")
            
            # Display metrics in a table
            st.subheader("Performance Metrics")
            metrics_df = pd.DataFrame(st.session_state.metrics)
            st.dataframe(metrics_df)
            
            # Create bar chart for model comparison
            st.subheader("Model Comparison")
            fig = px.bar(
                metrics_df, 
                barmode='group',
                color_discrete_sequence=px.colors.qualitative.G10,
                title="Performance Metrics Comparison"
            )
            st.plotly_chart(fig)
            
            # Confusion matrices
            st.subheader("Confusion Matrices")
            fig = plot_confusion_matrices(st.session_state.models, st.session_state.X_test, st.session_state.y_test)
            st.pyplot(fig)
            
            # ROC curves
            st.subheader("ROC Curves")
            fig = plot_roc_curves(st.session_state.models, st.session_state.X_test, st.session_state.y_test)
            st.pyplot(fig)
            
            st.success("Models evaluated successfully! You can now use them for spam classification.")

elif page == "Spam Classification":
    st.header("📧 Spam Classification")
    
    if not st.session_state.data_loaded:
        st.warning("Please upload a dataset first on the Data Upload & Analysis page.")
    elif st.session_state.models is None:
        st.warning("Please train models first on the Model Training & Evaluation page.")
    else:
        st.write("Use the trained models to classify emails as spam or ham.")
        
        # Text input for classification
        user_input = st.text_area("Enter the email text to classify:", height=200)
        
        # Model selection for classification
        model_name = st.selectbox(
            "Select model for classification",
            list(st.session_state.models.keys())
        )
        
        if st.button("Classify Email"):
            if user_input:
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
                        confidence = "Not available for this model"
                    
                    # Display result
                    st.subheader("Classification Result")
                    
                    if prediction == 'spam':
                        st.error(f"📛 This email is classified as **SPAM** with {confidence:.2f}% confidence.")
                    else:
                        st.success(f"✅ This email is classified as **HAM** (not spam) with {confidence:.2f}% confidence.")
                    
                    # Display feature importance if available
                    if model_name == "Random Forest":
                        st.subheader("Feature Importance Analysis")
                        
                        # Get feature names
                        feature_names = st.session_state.vectorizer.get_feature_names_out()
                        
                        # Get feature importances
                        importances = model.feature_importances_
                        
                        # Create DataFrame with feature names and importances
                        features_df = pd.DataFrame({
                            'feature': feature_names,
                            'importance': importances
                        })
                        
                        # Sort by importance
                        features_df = features_df.sort_values('importance', ascending=False).head(10)
                        
                        # Plot feature importances
                        fig = px.bar(
                            features_df,
                            x='importance',
                            y='feature',
                            orientation='h',
                            title="Top 10 Features",
                            color='importance',
                            color_continuous_scale='viridis'
                        )
                        st.plotly_chart(fig)
                    
                except Exception as e:
                    st.error(f"Error during classification: {str(e)}")
            else:
                st.warning("Please enter some text to classify.")
                
        # Example emails
        with st.expander("Try with example emails"):
            example_spam = "URGENT: You have WON a $1,000 Walmart gift card. Go to http://claim-prize.com to claim now before it expires!"
            example_ham = "Hi, just checking if we're still on for lunch tomorrow at 12:30pm? Let me know if you need to reschedule."
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Try Spam Example"):
                    st.session_state.example_text = example_spam
                    st.rerun()
            with col2:
                if st.button("Try Ham Example"):
                    st.session_state.example_text = example_ham
                    st.rerun()
            
            if 'example_text' in st.session_state:
                st.text_area("Example text:", st.session_state.example_text, height=100)

# Footer
st.markdown("---")
st.markdown("SpamShield - Protect your inbox from unwanted emails | Created with ❤️ using Streamlit")
