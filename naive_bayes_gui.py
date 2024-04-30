import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

spam_df = pd.read_csv("spam.csv")
spam_df['spam'] = spam_df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
x_train, x_test, y_train, y_test = train_test_split(spam_df.Message, spam_df.spam, test_size=0.25)
cv = CountVectorizer()
x_train_count = cv.fit_transform(x_train.values)
x_train_count.toarray()

# train model
model = MultinomialNB()
model.fit(x_train_count, y_train)

x_test_count = cv.transform(x_test)
print(model.score(x_test_count, y_test))

root = tk.Tk()
root.title("Naive Bayes Email Spam Filter")

def predict(text):
    x = cv.transform([text])
    res = model.predict(x)
    if res[0] == 0:
        print("Not spam!")
    else:
        print("Spam!")

def submit():
    text = text_input.get('1.0', 'end-1c')
    if text:
        predict(text)

text_input = tk.Text(root, height=10, width=50)
text_input.pack(pady=10)

submit_button = tk.Button(root, text="Submit", command=submit)
submit_button.pack()

root.mainloop()