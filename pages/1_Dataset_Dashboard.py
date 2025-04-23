import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
from collections import Counter

# Page configuration
st.set_page_config(
    page_title="SpamShield - Dataset Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.title("📊 Dataset Dashboard")
st.write("Explore the spam detection dataset and understand its characteristics")

# Get dataset from the session state
if 'dataset_info' in st.session_state and st.session_state.dataset_info is not None:
    df = st.session_state.dataset_info["df"]
    spam_count = st.session_state.dataset_info["spam_count"]
    ham_count = st.session_state.dataset_info["ham_count"]
    total_count = st.session_state.dataset_info["total_count"]
    
    # Create sections in tabs
    tab1, tab2, tab3 = st.tabs(["Dataset Overview", "Text Analysis", "Word Frequency"])
    
    with tab1:
        st.subheader("Dataset Statistics")
        
        # Top metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Emails", total_count)
        with col2:
            st.metric("Spam Emails", spam_count)
        with col3:
            st.metric("Ham Emails", ham_count)
        with col4:
            st.metric("Spam Ratio", f"{spam_count/total_count*100:.1f}%")
        
        # Dataset preview
        st.subheader("Sample Data")
        st.dataframe(df.sample(10))
        
        # Data distribution
        st.subheader("Data Distribution")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fig = px.pie(
                values=[spam_count, ham_count], 
                names=['Spam', 'Ham'], 
                title="Spam vs. Ham Distribution",
                color_discrete_sequence=['#ff6b6b', '#4ecdc4'],
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Calculate average email lengths
            df['text_length'] = df['text'].apply(len)
            
            avg_spam_length = df[df['label'] == 'spam']['text_length'].mean()
            avg_ham_length = df[df['label'] == 'ham']['text_length'].mean()
            
            fig = px.bar(
                x=['Spam', 'Ham'],
                y=[avg_spam_length, avg_ham_length],
                color=['Spam', 'Ham'],
                color_discrete_map={'Spam': '#ff6b6b', 'Ham': '#4ecdc4'},
                labels={'x': 'Email Type', 'y': 'Average Length (characters)'},
                title="Average Email Length by Type"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Text Length Analysis")
        
        # Add histogram of text lengths
        fig = px.histogram(
            df, 
            x='text_length', 
            color='label',
            nbins=50, 
            opacity=0.7,
            color_discrete_map={'spam': '#ff6b6b', 'ham': '#4ecdc4'},
            title="Distribution of Email Lengths",
            labels={'text_length': 'Email Length (characters)', 'count': 'Number of Emails'}
        )
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)
        
        # Email length statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Length Statistics: Spam Emails")
            spam_len = df[df['label'] == 'spam']['text_length']
            stats = {
                "Mean Length": spam_len.mean(),
                "Median Length": spam_len.median(),
                "Min Length": spam_len.min(),
                "Max Length": spam_len.max(),
                "Std Deviation": spam_len.std()
            }
            st.dataframe(pd.DataFrame(stats.items(), columns=["Statistic", "Value"]).set_index("Statistic"))
        
        with col2:
            st.subheader("Length Statistics: Ham Emails")
            ham_len = df[df['label'] == 'ham']['text_length']
            stats = {
                "Mean Length": ham_len.mean(),
                "Median Length": ham_len.median(),
                "Min Length": ham_len.min(),
                "Max Length": ham_len.max(),
                "Std Deviation": ham_len.std()
            }
            st.dataframe(pd.DataFrame(stats.items(), columns=["Statistic", "Value"]).set_index("Statistic"))
    
    with tab3:
        st.subheader("Word Frequency Analysis")
        
        # Function to count words
        @st.cache_data
        def get_word_counts(text_series, min_word_length=3, top_n=20):
            words = []
            for text in text_series:
                # Convert to lowercase and split
                clean_text = re.sub(r'[^\w\s]', '', text.lower())
                words.extend([word for word in clean_text.split() if len(word) >= min_word_length])
            
            # Count words
            word_counts = Counter(words)
            return word_counts.most_common(top_n)
        
        # Word count settings
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            min_word_length = st.slider("Minimum Word Length", 2, 10, 3)
        with col2:
            top_n = st.slider("Number of Top Words", 5, 50, 20)
        with col3:
            email_type = st.selectbox("Email Type", ["Spam", "Ham", "Both"])
        
        # Get the right subset of data
        if email_type == "Spam":
            text_data = df[df['label'] == 'spam']['text']
            color = '#ff6b6b'
            title = "Most Common Words in Spam Emails"
        elif email_type == "Ham":
            text_data = df[df['label'] == 'ham']['text']
            color = '#4ecdc4'
            title = "Most Common Words in Ham Emails"
        else:  # Both
            text_data = df['text']
            color = '#9e86d4'
            title = "Most Common Words in All Emails"
        
        # Get word counts
        word_counts = get_word_counts(text_data, min_word_length, top_n)
        
        # Create word count chart
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Bar chart
            word_df = pd.DataFrame(word_counts, columns=['word', 'count'])
            fig = px.bar(
                word_df,
                y='word',
                x='count',
                orientation='h',
                color_discrete_sequence=[color],
                title=title
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Word cloud
            try:
                wordcloud = WordCloud(
                    width=400, 
                    height=300,
                    background_color='white',
                    colormap='viridis',
                    max_words=100
                ).generate(' '.join([word for word, _ in word_counts]))
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Could not generate word cloud: {str(e)}")
                st.info("If you're running into errors, check if the wordcloud package is installed.")
                
else:
    st.error("Dataset information is not available. Please go back to the home page to load the data.")
    st.info("Navigate to the home page and ensure that the models are trained successfully.")