# simple-sentiment-analysis

## Description
The goal of this repository is to approach Natural Language Processing the simple way, and try explain the way it as I understand it.
I will be using **NLTK** library, for datasets and text preprocessing functions (text cleaning and preparation for classification), and **scikit learn** for classifying the polarity (positive, negative), in other words, creating the predictive ML model.
The dataset used is the **movie reviews**, containing 2000+ files separated by sentiments, 1000 of each.
This dataset is already categorized by polarity.

## Environment setup
- Install the required libraries `pip install -r requirements.txt`
- Download the **movie reviews**, **punkt** and **stopwords** NLTK datasets
    - Run the python interpreter
    - Import the NLTK library `import nltk`
    - Download the movie reviews dataset `nltk.download("movie_reviews")`
    - Download the word tokenizer dataset **punkt** `nltk.download("punkt")`
    - Download the stopwords list `nltk.download("stopwords")`


## Dataset preprocessing