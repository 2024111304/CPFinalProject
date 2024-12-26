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
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

data = pd.read_csv('Spam_SMS.csv')

# Drop duplicates and missing values, then describe the data
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)
data.describe()

#stopwords.word刪除
tokens = [word_tokenize(i) for i in data['Message']]
tkn = Tokenizer()
tkn.fit_on_texts(tokens)
stopwords_list = stopwords.words('english')
for i in range(len(tokens)):
    tokens[i] = [word for word in tokens[i] if word not in stopwords_list]
    tokens[i] = ' '.join(tokens[i])

data['Message'] = tokens
data['Message'] = data['Message'].str.lower()
data['Class'] = data['Class'].map({'ham': 0, 'spam': 1})

#########################畫出每個詞彙的TF-IDF值########################但我不知道為甚麼算出來的idf值長度是5159但是feature_name是9355,print(tfidf_matrix.shape)#output:(5159, 9355)
def tokenizer(text):
    return list(word_tokenize(text))

tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenizer,token_pattern=None, norm=None)

tf_mattrix=tfidf_vectorizer.fit_transform(data['Message'])
#print(tf_mattrix.shape[0])#output=5159
#取得詞語列表
feature_names = tfidf_vectorizer.get_feature_names_out()
#print(feature_names[0:3])
#print(feature_names.shape[0])#output=9355
tf_mattrix=tf_mattrix.toarray()
tf = pd.DataFrame(tf_mattrix, columns=feature_names)

tfidf_vectorizer.fit_transform(data['Message'])

idf_vector = tfidf_vectorizer.idf_
# 獲取每個詞彙的IDF值

idf = pd.DataFrame(idf_vector, index=feature_names, columns=["IDF"])
tfidf_matrix = tfidf_vectorizer.fit_transform(data['Message'])
idf.to_csv('IDF.csv')
idf1=pd.read_csv('IDF.csv')
#print(tfidf_matrix.shape)#(5159, 9355)
print(tfidf_matrix)

tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
# 獲取每個詞彙的TF-IDF值
tfidf_scores = tfidf_matrix.toarray()

# 繪製每個詞彙的TF-IDF值
plt.figure(figsize=(8, 8))
theta=np.linspace(0, 2 * np.pi, tfidf_scores.shape[0], endpoint=False)
r= tfidf_scores.mean(axis=1)
if len(theta) != len(r):
    raise ValueError(f"Theta and r must have the same length, but have lengths {len(theta)} and {len(r)}")
plt.polar(theta,tfidf_scores.mean(axis=1))
#plt.fill(np.linspace(0, 2 * np.pi, tfidf_scores.shape[0], endpoint=False), tfidf_scores.mean(axis=1), alpha=0.25)
#plt.xticks(np.linspace(0, 2 * np.pi, tfidf_scores.shape[0], endpoint=False), feature_names, rotation=90)
plt.title('TF-IDF Scores for Words')
plt.show()
#########################畫出每個詞彙的TF-IDF值########################


# Define all models and vectorizer

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
