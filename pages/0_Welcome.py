import streamlit as st

# Page configuration
st.set_page_config(
    page_title="SpamShield - Welcome",
    page_icon="👋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #4ecdc4, #2c3e50);
        padding: 20px;
        border-radius: 10px;
        color: white !important;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .header-text {
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    .feature-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        border-left: 5px solid #4ecdc4;
    }
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 10px;
        color: #4ecdc4;
    }
    .footer {
        text-align: center;
        margin-top: 30px;
        padding-top: 10px;
        border-top: 1px solid #ddd;
        color: #666;
        font-size: 0.8em;
    }
</style>
""", unsafe_allow_html=True)

# Header with custom styling
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.markdown('<h1 class="header-text">👋 Welcome to SpamShield</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="header-text">Advanced email spam detection using machine learning</h3>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Main content section with feature cards
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown('<div class="feature-icon">🧠</div>', unsafe_allow_html=True)
    st.markdown('### What is SpamShield?')
    st.write("""
    SpamShield is an advanced email spam detection application that uses machine learning 
    to protect your inbox from unwanted messages. Our intelligent algorithms analyze 
    email content and identify suspicious patterns to keep your inbox clean.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown('<div class="feature-icon">📊</div>', unsafe_allow_html=True)
    st.markdown('### Multiple Model Comparison')
    st.write("""
    Not all spam detection models are created equal. That's why SpamShield 
    uses multiple machine learning algorithms including Naive Bayes, SVM, 
    Random Forest, and Logistic Regression to provide the most accurate results.
    Compare their performance in real-time.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown('<div class="feature-icon">🔍</div>', unsafe_allow_html=True)
    st.markdown('### How Does It Work?')
    st.write("""
    1. Paste any email text on the home page
    2. Select which ML model to use
    3. Get instant classification results
    4. See detailed explanations of why the email was classified as spam or legitimate
    5. Explore the dataset to understand common spam patterns
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="feature-card">', unsafe_allow_html=True)
    st.markdown('<div class="feature-icon">📈</div>', unsafe_allow_html=True)
    st.markdown('### Data Visualization')
    st.write("""
    Explore our interactive data dashboard to understand common spam characteristics,
    word frequencies, and visualize how spam differs from legitimate emails. 
    View the performance metrics of each model to understand their strengths and weaknesses.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Call to action
st.markdown("""
<div style="background-color: #E8F5E9; padding: 20px; border-radius: 10px; text-align: center; margin-top: 20px;">
    <h2>Ready to stop spam in its tracks?</h2>
    <p style="font-size: 1.2rem;">Head to the home page and start classifying emails now!</p>
    <p>Use the sidebar navigation menu to explore all features of SpamShield.</p>
</div>
""", unsafe_allow_html=True)

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

# Center the logo 
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("assets/shield_logo.svg", width=150)

# Footer with styled text
st.markdown('<div class="footer">', unsafe_allow_html=True)
st.markdown("**SpamShield** - Email Spam Detection | Built with ❤️ using Streamlit and Machine Learning")
st.markdown('<p>© 2025 SpamShield | Protect your inbox like a boss</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)