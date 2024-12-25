import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv('Spam_SMS.csv')

# Drop duplicates and missing values, then describe the data
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)
data.describe()

# Change ham to 0 and spam to 1, then change all messages to lowercase
data['Class'] = data['Class'].map({'ham': 0, 'spam': 1})
data['Message'] = data['Message'].str.lower()

# Define all models and vectorizer
tkn = Tokenizer()
vec = CountVectorizer()

model_DT = DecisionTreeClassifier()
model_XGB = XGBClassifier()
model_MNB = MultinomialNB()
model_LR = LogisticRegression()
model_SVC = SVC()
model_RF = RandomForestClassifier()

# Fit the vectorizer and transform the data
X = vec.fit_transform(data['Message'])
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model_MNB.fit(X_train, y_train)
model_LR.fit(X_train, y_train)
model_SVC.fit(X_train, y_train)
model_RF.fit(X_train, y_train)
model_DT.fit(X_train, y_train)
model_XGB.fit(X_train, y_train)

# Get predictions
y_pred_MNB = model_MNB.predict(X_test)
y_pred_LR = model_LR.predict(X_test)
y_pred_SVC = model_SVC.predict(X_test)
y_pred_RF = model_RF.predict(X_test)
y_pred_DT = model_DT.predict(X_test)
y_pred_XGB = model_XGB.predict(X_test)

# Calculate confusion matrices
cm_MNB = confusion_matrix(y_test, y_pred_MNB)
cm_LR = confusion_matrix(y_test, y_pred_LR)
cm_SVC = confusion_matrix(y_test, y_pred_SVC)
cm_RF = confusion_matrix(y_test, y_pred_RF)
cm_DT = confusion_matrix(y_test, y_pred_DT)
cm_XGB = confusion_matrix(y_test, y_pred_XGB)

print("--- Confusion Matrices ---")
print(f"Multinomial Naive Bayes: \n{cm_MNB}")
print(f"Logistic Regression: \n{cm_LR}")
print(f"Support Vector Machine Classification: \n{cm_SVC}")
print(f"Random Forest Classification: \n{cm_RF}")
print(f"Decision Tree Classification: \n{cm_DT}")
print(f"XGBoost Classification: \n{cm_XGB}")

print(f"Accuracy: \n\
    Multinomial Naive Bayes: {model_MNB.score(X_test, y_test)*100:.2f}%, \n\
    Logistic Regression: {model_LR.score(X_test, y_test)*100:.2f}%, \n\
    Support Vector Machine Classification: {model_SVC.score(X_test, y_test)*100:.2f}%, \n\
    Random Forest Classification: {model_RF.score(X_test, y_test)*100:.2f}%, \n\
    Decision Tree Classification: {model_DT.score(X_test, y_test)*100:.2f}%, \n\
    XGBoost Classification: {model_XGB.score(X_test, y_test)*100:.2f}%")

def extract_spam_keywords(msg_vector, vectorizer, model):
    feature_names = vectorizer.get_feature_names_out()
    msg_features = msg_vector.toarray()[0]
    spam_keywords = [(feature_names[i], np.exp(model.feature_log_prob_[1][i]) * 100) for i, value in enumerate(msg_features)
                     if value > 0 and model.feature_log_prob_[1][i] > model.feature_log_prob_[0][i]]
    return spam_keywords

fig, axes = plt.subplots(3, 2, figsize=(15, 15))
axes = axes.flatten()

confusion_matrices = [cm_MNB, cm_LR, cm_SVC, cm_RF, cm_DT, cm_XGB]
titles = ["Multinomial Naive Bayes", "Logistic Regression", "Support Vector Machine", "Random Forest", "Decision Tree", "XGBoost"]

for ax, cm, title in zip(axes, confusion_matrices, titles):
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues', cbar=False)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.tight_layout()
plt.show()

while True:
    msg = input("Enter testing message (enter nothing to quit): ")
    if not msg:
        break
    msg = vec.transform([msg])
    pred = [model_MNB.predict(msg), model_LR.predict(msg), model_SVC.predict(msg), model_RF.predict(msg), model_DT.predict(msg), model_XGB.predict(msg)]
    model_names = ["NB", "LR", "SVC", "RF", "DT", "XGB"]
    pred_str = ""
    for i in range(len(pred)):
        if pred[i] == 1:
            pred_str += f"{model_names[i]} "
    if pred_str == "":
        pred_str = "None"
    spam_probability = sum(i[0] for i in pred)*100/len(pred)
    print(f"{spam_probability:.2f}% flagged as spam")
    print(f"Flagged by: {pred_str}")

    if spam_probability > 50:
        keywords = extract_spam_keywords(msg, vec, model_MNB)
        print("This message is flagged as SPAM.")
        print(f"Keywords indicating spam: ")
        for keyword in keywords:
            print(f"{keyword[0]} (Prob.: {keyword[1]:.3f}%)")
    else:
        print("This message is NOT flagged as spam.")