import pandas as pd
import numpy as np
import nltk
from keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv('./Spam_SMS.csv')

data.drop_duplicates(inplace=True)
data.dropna(inplace=True)
data.describe()

data['Class'] = data['Class'].map({'ham': 0, 'spam': 1})
data['Message'] = data['Message'].str.lower()

tkn = Tokenizer()
vec = CountVectorizer()

model_MNB = MultinomialNB()
model_LR = LogisticRegression()
model_SVC = SVC()
model_RF = RandomForestClassifier()

X = vec.fit_transform(data['Message'])
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_MNB.fit(X_train, y_train)
model_LR.fit(X_train, y_train)
model_SVC.fit(X_train, y_train)
model_RF.fit(X_train, y_train)

print(f"Accuracy: \n\
    Multinomial Naive Bayes: {model_MNB.score(X_test, y_test)*100:.2f}%, \n\
    Logistic Regression: {model_LR.score(X_test, y_test)*100:.2f}%, \n\
    Support Vector Machine Classification: {model_SVC.score(X_test, y_test)*100:.2f}%, \n\
    Random Forest Classification: {model_RF.score(X_test, y_test)*100:.2f}%")

while True:
    msg = input("Enter testing message (enter nothing to quit): ")
    if not msg:
        break
    msg = vec.transform([msg])
    pred = [model_MNB.predict(msg), model_LR.predict(msg), model_SVC.predict(msg), model_RF.predict(msg)]
    pred_results = ""
    if pred[0] == 1:
        pred_results += "NB "
    if pred[1] == 1:
        pred_results += "LR "
    if pred[2] == 1:
        pred_results += "SVC "
    if pred[3] == 1:
        pred_results += "RF"
    if not pred_results:
        pred_results = "None"
    print(f"{sum(i[0] for i in pred)*100/len(pred)}% flagged as spam")
    print(f"Flagged by: {pred_results}")