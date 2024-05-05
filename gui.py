# Use GUI to make predictions on new text
import tkinter as tk
from joblib import load

root = tk.Tk()
root.title("Ensemble Email Spam Filter")

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

    text_input.delete('1.0', tk.END)

text_input = tk.Text(root, height=10, width=50)
text_input.pack(pady=10)

submit_button = tk.Button(root, text="Submit", command=submit)
submit_button.pack()

result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()