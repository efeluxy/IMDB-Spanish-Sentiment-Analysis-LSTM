# IMDb Movie Reviews Sentiment Analysis with DNN
This repository contains a complete Data Science project focused on Natural Language Processing (NLP) and Deep Learning. The goal is to build a predictive model capable of classifying movie reviews in Spanish as either positive or negative.

### -- Project Overview --
The project implements an end-to-end pipeline, from Exploratory Data Analysis (EDA) to the deployment of a Deep Neural Network (DNN). It leverages a dataset of approximately 50,000 IMDb reviews, including both English and Spanish versions with their respective sentiment labels.

Key Objectives:
- Perform an in-depth EDA to understand data distribution and structure.
- Implement a robust NLP preprocessing pipeline for Spanish text.
- Develop and train a Deep Learning model to automate sentiment classification.

### -- Dataset Features --
The dataset consists of 5 main columns:
- Review Number: Unique identifier.
- Review (English): Original text.
- Review (Spanish): Translated text used for the model.
- Sentiment (English): Original label.
- Sentiment (Spanish): Target label for classification.

### -- Tech Stack --
- Language: Python
- Libraries:
    - Data Analysis: pandas, numpy
    - Visualization: matplotlib, seaborn
    - NLP: nltk, spacy, langdetect, emoji
    - Machine Learning: scikit-learn
    - Deep Learning: TensorFlow / Keras (Sequential API, Embedding, LSTM/Dense layers)

### -- Methodology --
1. Exploratory Data Analysis (EDA)
Comprehensive analysis of the dataset to ensure quality and balance:
- Class Distribution: Checking for balance between positive and negative sentiments.
- Text Metrics: Analyzing comment length and word frequency.
- Special Characters: Identifying emojis and punctuation patterns.

2. Preprocessing (NLP)
To prepare the text for the neural network, the following steps were performed:
- Cleaning: Removing unnecessary signs, special characters, and noise.
- Lemmatization & Tokenization: Reducing words to their base form and breaking text into tokens.
- Vectorization: Converting text into numerical format using Tokenizer and Padding to handle variable sequence lengths.

3. Model Architecture
A Deep Neural Network was designed to capture the semantic meaning of the reviews:
- Embedding Layer: For dense word representations.
- Hidden Layers: Utilizing Dense and/or LSTM layers to identify patterns in the text.
- Output Layer: A sigmoid-activated neuron for binary classification (Positive/Negative).

### -- Results --
The model was evaluated on a test set to verify its predictive power before final deployment. It includes a section for real-world testing with random samples to demonstrate its accuracy in classifying sentiments.
