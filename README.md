📩 Spam SMS Detection using Machine Learning
This project is a machine learning-based solution to classify SMS messages as Spam or Ham (Not Spam). It uses natural language processing (NLP) techniques for text preprocessing and a Naive Bayes classifier for prediction.

🧠 Overview
Objective: To develop a system that accurately detects spam messages.

Algorithm Used: Multinomial Naive Bayes

Libraries: Scikit-learn, NLTK, Pandas, NumPy, Matplotlib

📁 Project Structure
graphql
Copy
Edit
├── Spam_SMS_Detection.ipynb  # Main Jupyter notebook with code
├── spam.csv                  # Dataset containing SMS messages
└── README.md                 # Project documentation
📊 Dataset
Source: UCI SMS Spam Collection

2 columns:

v1: Label (ham/spam)

v2: Message text

🔍 Features & Workflow
Data Cleaning

Lowercasing

Removing punctuation

Stopword removal

Stemming

Feature Extraction

Bag of Words model using CountVectorizer

Model Building

Train/test split

Multinomial Naive Bayes classification

Evaluation

Accuracy score

Confusion matrix

Classification report

📈 Results
Achieved an accuracy of ~98%

High precision and recall for both spam and ham categories

▶️ How to Run
Clone this repository:

bash
Copy
Edit
git clone https://github.com/apurv77sharma/Spam-SMS-Detection.git
cd spam-sms-detection
Install required libraries:

bash
Copy
Edit
pip install -r requirements.txt
Open the Jupyter notebook:

bash
Copy
Edit
jupyter notebook Spam_SMS_Detection.ipynb
🛠 Tools & Technologies
Python

Jupyter Notebook

Scikit-learn

NLTK

Pandas & NumPy

📌 Future Work
Deploy the model using Flask or Streamlit

Add advanced NLP techniques like TF-IDF or word embeddings

Implement deep learning approaches

📬 Contact
Created by Apurv Sharma
📧 apurv77sharma@gmail.com
