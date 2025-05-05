import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import tkinter as tk
from tkinter import scrolledtext
from tkinter import messagebox

nltk.download('stopwords')

data = {
    "Question": [
        "What is the header file library in C?",
        "What is a comment in C and how to add it?",
        "What is the purpose of the sizeof operator in C?",
        "What is the difference between ++i and i++ in C?",
        "What is the purpose of break and continue statement in C"
    ],

    "Answer" : [
        "A header file library in C contains function declarations and macro definitions, included using #include for code finctionality (e.g. <stdio.h>)",
        "A comment in C is ignored by the compiler, used for explanations. Single line: // and Multi - line: /* comment */",
        "The sizeof operator gives the size (in bytes) of a data type or variable",
        "++i increments i before use, while i++ increments i after use",
        "Break exits a loop early, while continue skips the current loop iteration "
    ],

    "Advanced_Questions":[
        "What is recursion in C and how does it work? ",
        "What are the differeces between call by value and call by references?",
        "Explain the difference between static and dyanamic memory allocation in C",
        "What are the types of operations in C?",
        "Explain the difeerence between structures and unions in C"
    ]
}

df = pd.DataFrame(data)

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    if text:
        return " ".join([stemmer.stem(word) for word in text.lower().split() if word not in stop_words])
    return " "

df['Question'] = df ['Question'].apply (preprocess)

X_train, X_test, Y_train, Y_test = train_test_split(df['Question'], df['Answer'], test_size=0.2, random_state=42)

model = make_pipeline (TfidfVectorizer(ngram_range=(1,2)), MultinomialNB(alpha = 0.1))
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)
print(f"MOdel Accuracy: {accuracy_score(Y_test, Y_pred): .2f}")

class programminghelperapp:
    def __init__(self,root,model):
        self.root = root
        self.model = model
        self.advanced_index = 0
        self.basic_answered = 0

        self.root.title ("Welcome to Programming Helper 👩‍💻")
        self.root.geometry("900x700")
        self.root.config(bg ="#f4f8fb")

        self.header_label = tk.Label(root, text="Ask your programming Questions", font=("Arial",18,"bold"), bg="#f4f8fb", fg="black")
        self.header_label.pack(pady=20)

        self.input_frame = tk.Frame(root, bg="#f0f0f0")
        self.input_frame.pack(pady=10)

        self.input_label= tk.Label(self.input_frame, text="Question:",font=("Arial",12), bg="#ffffff",fg="#333")
        self.input_label.grid(row=0, column=0, padx=10)

        self.input_text = tk.Entry(self.input_frame, width=50, font=("Arial",12),bg="#ffffff", fg="#333")
        self.input_text.grid(row=0, column=1, padx=10)

        self.ask_button = tk.Button(self.input_frame, text="Submit", command=self.get_answer, font=("Arial",12), bg="#4CAF50", fg="white")
        self.ask_button.grid(row=0, column=2, padx=10)

        self.output_frame = tk.Frame(root, bg="#f0f0f0")
        self.output_frame.pack(pady=30)

        self.output_text = scrolledtext.ScrolledText(self.output_frame, width=80, height = 30, font=("Arial",12), wrap=tk.WORD, bg="#ffffff", fg="#333")
        self.output_text.pack()

    def get_answer(self):
        user_input = self.input_text.get().strip()

        if user_input:
            processed_input = preprocess(user_input)

            predicted_probabilities = self.model.predict_proba([processed_input])[0]
            max_probability = np.max(predicted_probabilities)
            predicted_answer = self.model.predict([processed_input])[0]

            confidence_threshold = 0.5

            self.output_text.insert(tk.END, f"Q: {user_input}\n\n")

            if max_probability> confidence_threshold:
                self.output_text.insert(tk.END, f"A: {predicted_answer}\n\n")
            else:
                self.output_text.insert(tk.END, f"A: Have a doubt. Please get help from https://www.programiz.com/c-programming.\n\n")

            self.basic_answered += 1

            if self.basic_answered>=5:
                self.ask_Advanced_Questions()
            
            self.input_text.delete(0, tk.END)
        else:
            self.output_text.insert(tk.END, "Please enter question.\n\n")
    
    def ask_Advanced_Questions(self):
        Answer = messagebox.askyesno("Advanced Questions", "Would you like to get advanced questions?")

        if Answer:
            if self.advanced_index<len(df['Advanced_Questions']):
                Advanced_Questions = df['Advanced_Questions'][self.advanced_index]
                self.output_text.insert(tk.END, f"\n Advanced Question:{Advanced_Questions}\n")
                self.ask_for_answer_or_help()
            else:
                self.output_text.insert(tk.END, "\n No more advanced questions available\n")
        else:
            self.output_text.insert(tk.END, "\n You choose not to answer advanced questions. Thank you !\n")

    def ask_for_answer_or_help(self):
        Answer = messagebox.askyesno  ("Answer or Help", "Do you know the answer?")

        if Answer:
            self.output_text.insert(tk.END, "Please enter your answer..\n")
        else:
            self.output_text.insert(tk.END, "You can get help from: https://www.programiz.com/c-programming\n")

        self.advanced_index += 1

if __name__ == "__main__":
    root = tk.Tk()
    app = programminghelperapp(root,model)
    root.mainloop()
