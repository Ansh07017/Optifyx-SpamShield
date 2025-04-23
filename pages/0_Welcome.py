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
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Roboto:wght@300;400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    body {
        background-image: url("assets/backgrounds/shield_pattern.svg");
        background-attachment: fixed;
        background-size: cover;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        letter-spacing: -0.01em;
        color: #2c3e50;
    }
    
    p, li, div {
        font-family: 'Roboto', sans-serif;
        font-weight: 400;
        font-size: 1.05rem;
        line-height: 1.5;
        color: #000000;
    }
    
    .main-header {
        background: linear-gradient(90deg, #4ecdc4, #2c3e50);
        padding: 20px;
        border-radius: 10px;
        color: white !important;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: url("assets/backgrounds/wave_pattern.svg");
        background-size: cover;
        opacity: 0.3;
        z-index: 0;
    }
    
    .header-text {
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        position: relative;
        z-index: 1;
        font-family: 'Poppins', sans-serif;
        letter-spacing: 0.02em;
    }
    
    .feature-card {
        background-color: rgba(248, 249, 250, 0.95);
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        margin-bottom: 24px;
        border-left: 5px solid #4ecdc4;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.1);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 15px;
        color: #4ecdc4;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .footer {
        text-align: center;
        margin-top: 30px;
        padding: 20px;
        border-top: 1px solid #ddd;
        color: #666;
        font-size: 0.9em;
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        font-family: 'Poppins', sans-serif;
    }
    
    /* Better link styling */
    a {
        color: #4ecdc4;
        text-decoration: none;
        font-weight: 500;
    }
    
    a:hover {
        color: #3dafa7;
        text-decoration: underline;
    }
    
    /* Fix headers */
    h1, h2, h3 {
        margin-top: 0.5em;
        margin-bottom: 0.5em;
    }
    
    /* Call to action styling */
    .cta-container {
        background-color: rgba(232, 245, 233, 0.9);
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        margin-top: 24px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        border-top: 4px solid #4CAF50;
    }
    
    .cta-container h2 {
        color: #2E7D32;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        margin-bottom: 12px;
    }
    
    .cta-container p {
        font-family: 'Roboto', sans-serif;
        font-size: 1.1rem;
        margin-bottom: 12px;
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
<div class="cta-container">
    <h2>Ready to stop spam in its tracks?</h2>
    <p>Head to the home page and start classifying emails now!</p>
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