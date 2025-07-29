Email/SMS Spam Classifier
This project is a machine learning-based application designed to classify emails or SMS messages as either "Spam" or "Not Spam" (often referred to as "Ham"). It utilizes Natural Language Processing (NLP) techniques to process the text data and a trained Naive Bayes model to make predictions. The application is built with Streamlit, providing a simple and interactive web interface for users to input messages and get instant classification results.

📋 Features
Text Preprocessing: Converts text to lowercase, tokenizes sentences, removes special characters, stop words, and punctuation.

Stemming: Reduces words to their root form to simplify the vocabulary.

TF-IDF Vectorization: Transforms the processed text into numerical feature vectors.

Machine Learning Model: Employs a Multinomial Naive Bayes classifier for prediction.

Web Interface: A user-friendly interface built with Streamlit to input messages and view the prediction.

💻 Technologies Used
Python: The core programming language.

Scikit-learn: For machine learning algorithms and vectorization.

NLTK (Natural Language Toolkit): For text preprocessing tasks like tokenization, stop word removal, and stemming.

Streamlit: To create the interactive web application.

Pandas & NumPy: For data manipulation and numerical operations.

Jupyter Notebook: For model development and experimentation.

Pickle: For saving and loading the trained model and vectorizer.

⚙️ Setup and Installation
To run this project on your local machine, please follow these steps:

Clone the Repository

git clone https://github.com/Harshit-Tiwari-29/email-sms-spam-classifier.git
cd email-sms-spam-classifier

Create a Virtual Environment (Recommended)

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

Install Dependencies
Make sure you have a requirements.txt file with the necessary libraries. If not, create one and add the following content:

streamlit
scikit-learn
nltk
pandas
numpy

Then, install the packages:

pip install -r requirements.txt

Download NLTK Data
You will need to download the 'punkt' and 'stopwords' packages from NLTK. Run the following command in a Python interpreter:

import nltk
nltk.download('punkt')
nltk.download('stopwords')

Run the Streamlit App
Once the dependencies are installed, you can run the application:

streamlit run app.py

The application should now be running and accessible in your web browser.

🚀 Usage
Open the web application in your browser (usually at http://localhost:8501).

In the text input box labeled "Enter the message", type or paste the email or SMS you want to classify.

Click the "Predict" button.

The application will display the prediction: "Spam" or "Not Spam".

📂 File Descriptions
app.py: The main Python script that runs the Streamlit web application.

Email_SMS_spam_detection.ipynb: A Jupyter Notebook containing the complete process of data cleaning, preprocessing, model training, and evaluation.

vectorizer.pkl: A serialized Python object containing the fitted TF-IDF vectorizer.

model.pkl: A serialized Python object containing the trained Multinomial Naive Bayes model.

requirements.txt: A list of Python libraries required to run the project.

🤖 Model Information
The classification model is a Multinomial Naive Bayes classifier, which is well-suited for text classification tasks. The text data is converted into numerical vectors using the Term Frequency-Inverse Document Frequency (TF-IDF) method, with a maximum of 3000 features. The model was trained on a dataset of SMS messages and achieved a high precision score, making it effective at identifying spam messages correctly.

🤝 Contributing
Contributions are welcome! If you have any suggestions for improvements or find any issues, please feel free to open an issue or submit a pull request.
