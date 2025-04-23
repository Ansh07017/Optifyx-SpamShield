import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc, confusion_matrix

# Page configuration
st.set_page_config(
    page_title="SpamShield - Model Performance",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.title("📈 Model Performance Analysis")
st.write("Compare and analyze the performance of different machine learning models for spam detection")

# Get model data from the session state
if ('models' in st.session_state and st.session_state.models is not None and
    'metrics' in st.session_state and st.session_state.metrics is not None and
    'X_test' in st.session_state and st.session_state.X_test is not None and
    'y_test' in st.session_state and st.session_state.y_test is not None):
    
    models = st.session_state.models
    metrics = st.session_state.metrics
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    
    # Convert metrics to dataframe
    metrics_df = pd.DataFrame(metrics).T  # Transpose to get model names as index
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Model Comparison", "Confusion Matrices", "ROC Curves"])
    
    with tab1:
        st.subheader("Model Performance Metrics")
        
        # Display metrics table
        st.dataframe(metrics_df.style.highlight_max(axis=0, color='#4CAF50').format("{:.4f}"))
        
        # Metric selection for detailed view
        selected_metric = st.selectbox(
            "Select metric for comparison:",
            ["Accuracy", "Precision", "Recall", "F1 Score"]
        )
        
        # Create bar chart for the selected metric
        fig = px.bar(
            metrics_df,
            y=selected_metric,
            color=selected_metric,
            color_continuous_scale=px.colors.sequential.Viridis,
            text_auto='.4f',
            title=f"{selected_metric} Comparison Across Models"
        )
        fig.update_layout(
            xaxis_title="Model",
            yaxis_title=selected_metric,
            yaxis_range=[metrics_df[selected_metric].min() * 0.95, min(1.0, metrics_df[selected_metric].max() * 1.05)]
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Multi-metric comparison
        st.subheader("Multi-Metric Comparison")
        
        # Radar chart for comparing all metrics across models
        metrics_list = ["Accuracy", "Precision", "Recall", "F1 Score"]
        
        fig = go.Figure()
        
        for model_name in metrics_df.index:
            fig.add_trace(go.Scatterpolar(
                r=[metrics_df.loc[model_name, metric] for metric in metrics_list],
                theta=metrics_list,
                fill='toself',
                name=model_name
            ))
            
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[metrics_df[metrics_list].min().min() * 0.9, 1]
                )
            ),
            title="Model Performance Radar Chart"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Confusion Matrices")
        
        # Model selection for confusion matrix
        selected_model = st.selectbox(
            "Select model:",
            list(models.keys())
        )
        
        if selected_model:
            # Get confusion matrix
            y_pred = models[selected_model].predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            # Create confusion matrix display
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Custom color mapping
            cax = ax.matshow(cm, cmap=plt.cm.Blues)
            plt.colorbar(cax)
            
            # Set labels
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title(f'Confusion Matrix - {selected_model}')
            
            # Set x and y axis tick labels
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Ham', 'Spam'])
            ax.set_yticklabels(['Ham', 'Spam'])
            
            # Add text annotations
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    text_color = 'white' if cm[i, j] > cm.max() / 2. else 'black'
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=text_color)
            
            st.pyplot(fig)
            
            # Calculate and display additional metrics
            true_negative = cm[0, 0]
            false_positive = cm[0, 1]
            false_negative = cm[1, 0]
            true_positive = cm[1, 1]
            
            accuracy = (true_positive + true_negative) / cm.sum()
            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
            recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Accuracy", f"{accuracy:.4f}")
            with col2:
                st.metric("Precision", f"{precision:.4f}")
            with col3:
                st.metric("Recall", f"{recall:.4f}")
            with col4:
                st.metric("F1 Score", f"{f1:.4f}")
            with col5:
                st.metric("Specificity", f"{specificity:.4f}")
            
            # Show interpretation
            st.markdown("""
            ### Interpretation
            
            - **True Negative (TN):** Correctly identified legitimate emails
            - **False Positive (FP):** Legitimate emails incorrectly classified as spam
            - **False Negative (FN):** Spam emails incorrectly classified as legitimate
            - **True Positive (TP):** Correctly identified spam emails
            """)
            
            # Analysis of misclassifications 
            st.write(
                f"This model correctly identified {true_negative} legitimate emails and {true_positive} spam emails. "
                f"However, it misclassified {false_positive} legitimate emails as spam and "
                f"missed {false_negative} spam emails that were incorrectly classified as legitimate."
            )
    
    with tab3:
        st.subheader("ROC Curves")
        
        # Plot ROC curves for all models
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Colors for different models
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        
        for (model_name, model), color in zip(models.items(), colors):
            try:
                # Get prediction probabilities
                y_prob = model.predict_proba(X_test)[:, 1]
                
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                
                # Plot ROC curve
                ax.plot(fpr, tpr, color=color, lw=2,
                       label=f'{model_name} (AUC = {roc_auc:.3f})')
            except Exception as e:
                st.warning(f"Could not plot ROC curve for {model_name}: {str(e)}")
        
        # Add random guess line
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
        
        # Set labels and title
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curves')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # ROC curve explanation
        st.markdown("""
        ### Understanding ROC Curves
        
        The Receiver Operating Characteristic (ROC) curve plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold settings.
        
        - **Area Under the Curve (AUC):** A measure of model performance. Higher values (closer to 1.0) indicate better performance.
        - **True Positive Rate (TPR):** Also known as Recall, it measures the proportion of spam emails correctly identified.
        - **False Positive Rate (FPR):** The proportion of legitimate emails incorrectly classified as spam.
        
        For spam detection, a good model should have a high AUC value, maximizing the detection of spam while minimizing the incorrect classification of legitimate emails as spam.
        """)
        
else:
    st.error("Model information is not available. Please go back to the home page to train the models.")
    st.info("Navigate to the home page and ensure that the models are trained successfully.")