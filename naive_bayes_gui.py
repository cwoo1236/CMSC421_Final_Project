# import packages
# https://www.youtube.com/watch?v=2sXAYoPIz3A
# https://towardsdatascience.com/na%C3%AFve-bayes-spam-filter-from-scratch-12970ad3dae7
# https://github.com/makispl/SMS-Spam-Filter-Naive-Bayes/blob/master/SMS_Spam_Filtering_Naive_Bayes.ipynb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Importing NLTK for natural language processing
import nltk
from nltk.corpus import stopwords    # For stopwords


# Downloading NLTK data
nltk.download('stopwords')   # Downloading stopwords data
nltk.download('punkt')       # Downloading tokenizer data

# Importing WordCloud for text visualization
from wordcloud import WordCloud

import matplotlib.pyplot as plt  # For data visualization

# import data from https://www.kaggle.com/datasets/mfaisalqureshi/spam-email?resource=download
spam_df = pd.read_csv("spam.csv")