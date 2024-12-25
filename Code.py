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

data = pd.read_csv(r"C:\Users\user\程設\期末專題\CPFinalProject\Spam_SMS.csv")

data.drop_duplicates(inplace=True)
data.dropna(inplace=True)
data.describe()

data['Class'] = data['Class'].map({'ham': 0, 'spam': 1})
data['Message'] = data['Message'].str.lower()

tkn = Tokenizer()
vec = CountVectorizer()

model_DT = DecisionTreeClassifier()
model_XGB = XGBClassifier()
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

# Get predictions
y_pred_MNB = model_MNB.predict(X_test)
y_pred_LR = model_LR.predict(X_test)
y_pred_SVC = model_SVC.predict(X_test)
y_pred_RF = model_RF.predict(X_test)

# Calculate confusion matrices
cm_MNB = confusion_matrix(y_test, y_pred_MNB)
cm_LR = confusion_matrix(y_test, y_pred_LR)
cm_SVC = confusion_matrix(y_test, y_pred_SVC)
cm_RF = confusion_matrix(y_test, y_pred_RF)

print("--- Confusion Matrices ---")
print(f"Multinomial Naive Bayes: \n{cm_MNB}")
print(f"Logistic Regression: \n{cm_LR}")
print(f"Support Vector Machine Classification: \n{cm_SVC}")
print(f"Random Forest Classification: \n{cm_RF}")

print(f"Accuracy: \n\
    Multinomial Naive Bayes: {model_MNB.score(X_test, y_test)*100:.2f}%, \n\
    Logistic Regression: {model_LR.score(X_test, y_test)*100:.2f}%, \n\
    Support Vector Machine Classification: {model_SVC.score(X_test, y_test)*100:.2f}%, \n\
    Random Forest Classification: {model_RF.score(X_test, y_test)*100:.2f}%")

# ------------------------------
# 新增：關鍵字分析函數
# ------------------------------
def extract_spam_keywords(msg, vectorizer, model):
    """
    提取導致訊息被標記為 spam 的關鍵字
    """
    msg_vector = vectorizer.transform([msg])
    feature_names = vectorizer.get_feature_names_out()
    msg_features = msg_vector.toarray()[0]
    spam_keywords = [feature_names[i] for i, value in enumerate(msg_features) 
                     if value > 0 and model.feature_log_prob_[1][i] > model.feature_log_prob_[0][i]]
    return spam_keywords

# ------------------------------
# 測試輸入訊息
# ------------------------------
while True:
    msg = input("Enter testing message (enter nothing to quit): ")
    if not msg:
        break
    msg_vector = vec.transform([msg])
    pred = [model_MNB.predict(msg_vector), model_LR.predict(msg_vector), model_SVC.predict(msg_vector), model_RF.predict(msg_vector)]
    spam_probability = sum(i[0] for i in pred) * 100 / len(pred)
    print(f"{spam_probability}% flagged as spam")
    
    # 如果是 spam，顯示關鍵字
    if spam_probability > 50:
        keywords = extract_spam_keywords(msg, vec, model_MNB)
        print("This message is flagged as SPAM.")
        print(f"Keywords indicating spam: {', '.join(keywords) if keywords else 'No specific keywords detected'}")
    else:
        print("This message is NOT flagged as spam.")
