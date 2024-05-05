# Use GUI to make predictions on new text
from tkinter import *
from joblib import load

root = Tk()
root.title("Ensemble Email Spam Filter")
root.configure(bg='white')

# exclude KNN due to poor accuracy
classifiers = dict()
classifier_names = ["LR", "SVC", "NB", "DT", "RF"]
for name in classifier_names:
    classifiers[name] = load(f"models/{name}_model.joblib")

tfid = load("models/tfid.joblib")

def ensemble_predict(text):
    x = tfid.transform([text]).toarray()
    predictions = []

    for name, model in classifiers.items():
        pred = model.predict(x)
        predictions.append(pred[0])
    
    print(predictions)

    # we have 6 models; err on the side of more false negatives
    if predictions.count(0) >= 3:
        result_label.config(text=f"Not spam! -- {text}")
    else:
        result_label.config(text=f"Spam! -- {text}")

def submit():
    text = text_input.get('1.0', 'end-1c')
    if text:
        ensemble_predict(text)

    text_input.delete('1.0', END)

title = Label(root, text="Email Spam Filter", font=("HelvLight", 20), pady=10, bg='white')
title.pack()

frame = Frame(root, bg='white')
frame.pack(fill="both", expand=True, padx=10, pady=10)

text_input = Text(frame, height=2, width=50, relief=FLAT, bg='#dedede')
text_input.pack(pady=10)

submit_button = Button(frame, text="Submit", relief=FLAT, command=submit, bg="#4287f5", fg="white", pady=5)
submit_button.pack()

result_label = Label(frame, text="", bg='white')
result_label.pack()

root.mainloop()