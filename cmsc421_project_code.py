# Table of Contents
###1. Introduction
###2. Data Collection & Parsing
###3. Mode Building
###4. Hypothesis Testing
###5. Conclusion

# 1. Introduction

##Background

## Examples

# 2. Data Collection & Parsing

# @title Libraries & Resources
# https://www.youtube.com/watch?v=2sXAYoPIz3A
# https://towardsdatascience.com/na%C3%AFve-bayes-spam-filter-from-scratch-12970ad3dae7
# https://github.com/makispl/SMS-Spam-Filter-Naive-Bayes/blob/master/SMS_Spam_Filtering_Naive_Bayes.ipynb

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing NLTK for natural language processing
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

# Downloading NLTK data
nltk.download('stopwords')   # Downloading stopwords data
nltk.download('punkt')       # Downloading tokenizer data

from sklearn.metrics import accuracy_score, precision_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.ensemble import BaggingClassifier
#from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.ensemble import GradientBoostingClassifier
#from xgboost import XGBClassifier

from joblib import dump

# Importing WordCloud for text visualization
from wordcloud import WordCloud

# @title Reading in the Data
# import data from https://www.kaggle.com/datasets/mfaisalqureshi/spam-email?resource=download
csv_path = "enron_spam_data.csv"
df = pd.read_csv(csv_path, on_bad_lines='skip')

# @title Preliminary Parsing
# Removing rows with NaN values in the Message and Spam/Ham columns
df = df.dropna(subset=['Message'])
df = df.dropna(subset=['Spam/Ham'])

# Quantify Spam and Ham
df['Spam/Ham'] = df['Spam/Ham'].apply(lambda x: 1 if x == 'spam' else 0)

# @title Data Preprocessing

# Initialize NLTK's stopwords and Porter Stemmer
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
porter_stemmer = PorterStemmer()

# Function for text preprocessing
def preprocess_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    # Remove special characters, stopwords, and punctuation
    filtered_tokens = [token for token in tokens if token.isalnum() and token not in stop_words and token not in string.punctuation]

    stemmed_tokens = [porter_stemmer.stem(token) for token in filtered_tokens]
    processed_text = ' '.join(stemmed_tokens)

    return processed_text

# Adding a new column with processed data
df["Transformed Message"] = df["Message"].apply(preprocess_text)
df

# @title Spam Word Cloud
wc = WordCloud(width = 500, height = 500, min_font_size = 10, background_color = 'white')
spam_wc = wc.generate(df[df['Spam/Ham'] == 1]['Transformed Message'].str.cat(sep = " "))
plt.figure(figsize = (15,6))
plt.imshow(spam_wc)
plt.show()

"""# 3. Modeling"""

# @title Model Building
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

cv = CountVectorizer()
tfid = TfidfVectorizer(max_features = 3000)

x = tfid.fit_transform(df['Transformed Message']).toarray()
y = df['Spam/Ham'].values

dump(tfid, "models/tfid.joblib")

x_train, x_test , y_train, y_test = train_test_split(x,y,test_size = 0.20, random_state = 2)
classifiers = {"LR": LogisticRegression(solver="liblinear", penalty="l1"),
               "SVC": SVC(kernel= "sigmoid", gamma  = 1.0),
               "NB": MultinomialNB(),
               "DT": DecisionTreeClassifier(max_depth = 5),
               "KNN": KNeighborsClassifier(),
               "RF": RandomForestClassifier(n_estimators = 50, random_state = 2)}

def train_classifier(clfs, X_train, y_train, X_test, y_test):
    clfs.fit(X_train,y_train)
    y_pred = clfs.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    return accuracy , precision

# @title Evaluating Models
for name, classifier in classifiers.items():
    current_accuracy, current_precision = train_classifier(classifier, x_train, y_train, x_test, y_test)
    dump(classifier, f'models/{name}_model.joblib')
    print("\nFor: ", name)
    print("Accuracy: ", current_accuracy)
    print("Precision: ", current_precision)

"""Other resources:
https://www.kaggle.com/code/zabihullah18/email-spam-detection

https://www.kaggle.com/code/zabihullah18/email-spam-detection/input
- spam assassin: https://www.kaggle.com/code/zabihullah18/email-spam-detection/input?select=completeSpamAssassin.csv
- enron dataset: https://www.kaggle.com/code/zabihullah18/email-spam-detection/input?select=emails.csv

https://www.kaggle.com/datasets/marcelwiechmann/enron-spam-data?select=enron_spam_data.csv


Works Cited Links:
- https://mailtrap.io/blog/spam-filters/
"""

