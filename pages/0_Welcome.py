import streamlit as st

# Page configuration
st.set_page_config(
    page_title="SpamShield - Welcome",
    page_icon="👋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.title("👋 Welcome to SpamShield")

# Main content
st.markdown("""
## About SpamShield

SpamShield is an advanced email spam detection application that uses machine learning 
to protect your inbox from unwanted messages. Our application provides:

### Key Features

- **Email Classification**: Analyze emails to determine if they're spam or legitimate
- **Multiple Models**: Compare different machine learning algorithms for best results
- **Detailed Insights**: Understand why an email is classified as spam or legitimate
- **Data Visualization**: Explore the dataset and understand spam email patterns

### How to Use

1. **Email Detection**: On the home page, paste any email text to check if it's spam
2. **Dataset Exploration**: View the Dataset Dashboard to explore spam email characteristics
3. **Model Comparison**: Check the Model Performance page to see how different algorithms perform

### Get Started

Navigate using the sidebar menu to explore the different features of our application.

""")

# Show information on model types
with st.expander("Learn about our machine learning models", expanded=False):
    st.markdown("""
    ### Machine Learning Models Used
    
    SpamShield uses multiple machine learning models to classify emails:
    
    1. **Naive Bayes**: Fast and effective for text classification
    2. **Support Vector Machine (SVM)**: Great for high-dimensional data like text
    3. **Random Forest**: Ensemble method with good explanatory power
    4. **Logistic Regression**: Simple but effective probabilistic classifier
    
    Each model has different strengths, which is why we provide all of them for your use.
    """)

# Simple visualization of workflow
st.image("assets/shield_logo.svg", width=200)

# Footer
st.write("---")
st.caption("SpamShield - Email Spam Detection | Built with Streamlit and Machine Learning")