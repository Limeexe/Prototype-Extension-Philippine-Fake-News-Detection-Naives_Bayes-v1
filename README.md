# Prototype-Extension-Philippine-Fake-News-Detection-Naives_Bayes-v1

This is a Python-based Chrome extension designed to detect fake news by analyzing text content created for the purpose of the research thesis entitled "**Development and Prototype Implementation of a Browser Extension for Fake News Detection in Philippine News Using Natural Language Processing Algorithms**" in Computer Science Thesis 1 & 2 subject in Bachelor of Computer Science in Camarines Sur Polytechnic Colleges. The extension uses machine learning models to determine the credibility of news articles.


This Chrome extension uses a machine learning model based on the Naive Bayes algorithm to detect fake news. It processes text content from web pages or documents, analyzing their credibility and detecting suspicious elements.



## Installation

1. **Prerequisites**
   - Python 3.8 or higher (for running Flask server)
   - Required Libraries:
     - `node`
     - `python3`
     - `flask`
     - `pandas`
     - `numpy`
     - `joblib`
     - `sumy`
     - `flask`

2. **Steps to Install**
   ```bash
   # Install required Python dependencies
   python3 -m pip install flask pandas numpy joblib sumy nltk flask flask-cors

## Usage

1. Open a web page or document with suspected fake news content
2. Create a new tab (Ctrl + T)
3. Click the extension icon and press the chrome extension

The extension will display:
- Credibility score for the content
- Highlighted credible and suspicious words
- Tips section with detection tips

## Features

1. Text Preprocessing: 
   - Lowercase conversion
   - Removed punctuation and digits
   - Tokenization
   - Stopword removal
   - Lemmatization

2. Credibility Logic:
   - Uses TF-IDF vectorizer to convert text into numerical features
   - Predicts probability of content being suspicious using calibrated Naive Bayes
   - Updates credibility based on confidence level (<= 0.8: Credible, >= 0.8: Suspicious)

3. Influence Word Detection:
   - Identifies important words affecting the prediction
   - Highlights credible and suspicious words with different colors

4. Progress Bar:
   - Visual representation of processing time using LSA summarizer

5. Tip Display:
   - Shows additional detection tips for spotting fake news

## Technical Details

* Uses Naive Bayes algorithm with TF-IDF vectorizer
* Processes text in English only
* Requires PTLB (punkt, stopword, wordnet) NLTK packages
* Runs locally using Flask backend



