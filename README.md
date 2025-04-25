# 📬 Optifyx SpamShield – Email Spam Classifier

**Optifyx SpamShield** is a lightweight and interactive web application built using **Streamlit** that detects whether an email is **Spam** or **Legitimate (Ham)**. It leverages multiple machine learning models to analyze and classify messages based on the contents of emails.

## 🧠 Models Used

This app allows users to choose from the following machine learning models for spam classification:

1. **Logistic Regression**  
   A linear model that estimates the probability of an email being spam or not.

2. **K-Nearest Neighbors (KNN)**  
   Classifies emails based on the most similar (nearest) training samples.

3. **Support Vector Machine (SVM)**  
   Constructs a hyperplane to optimally separate spam and ham messages.

4. **Decision Tree**  
   Uses a tree structure to split messages based on key words/features.

5. **Random Forest**  
   An ensemble of decision trees that votes for the final prediction.

6. **Naive Bayes**  
   A probabilistic model particularly effective for text classification tasks.

---

## 📦 Dataset

The app uses the **spam.csv** dataset, which contains labeled SMS/email messages. It includes:

- **Label**: 'spam' or 'ham'
- **Message**: The actual email/text content

This dataset is commonly used for binary classification tasks in NLP.

---

## 🖥️ How to Run the App

1. Clone this repository:
   ```bash
   git clone https://github.com/Ansh07017/Optifyx-SpamShield.git
   cd Optifyx-SpamShield
2. pip install -r requirements.txt
3. streamlit run app.py


