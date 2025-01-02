import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from textblob import TextBlob
from progress.bar import Bar
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 讀取兩個檔案，指定編碼為 'ISO-8859-1'
data1 = pd.read_csv('Spam_SMS.csv', encoding='ISO-8859-1')
data2 = pd.read_csv('Spam_SMS_ChatGPT.csv', encoding='ISO-8859-1')

# 合併兩個資料集
data = pd.concat([data1, data2], ignore_index=True)

# Drop duplicates and missing values, then describe the data
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)

# 確保所有值都是字串
data['Message'] = data['Message'].astype(str)

#stopwords.word刪除
tokens = [word_tokenize(i) for i in data['Message']]
stopwords_list = stopwords.words('english')
processed_tokens = []
for  token_list in tokens:
    filtered_tokens = [word for word in token_list if word.lower() not in stopwords_list]
    if filtered_tokens:  # 確保不會移除所有的詞彙
        processed_tokens.append(' '.join(filtered_tokens))
    else:
        processed_tokens.append('dummy')  # 如果全是停用詞，保留一個佔位符詞彙

data['Message'] = processed_tokens

'''data['Message'] = tokens'''
data['Message'] = data['Message'].str.lower()
data['Class'] = data['Class'].map({'ham': 0, 'spam': 1})

# 初始化 CountVectorizer 並擬合數據
vec = CountVectorizer()
X = vec.fit_transform(data['Message'])
y = data['Class'].values

'''
#########################畫出每個詞彙的TF-IDF值########################
documents = data['Message'].values

# 初始化 TfidfVectorizer 並擬合數據
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# 獲取 IDF 值
idf_values = vectorizer.idf_
feature_names = vectorizer.get_feature_names_out()
# 將 IDF 值和詞彙名稱組合在一起
idf_items = list(zip(feature_names, idf_values))

# 按照 IDF 值排序，取前 20 個最低的
idf_items_sorted_lowest20 = sorted(idf_items, key=lambda x: x[1])[:20]
idf_items_sorted_highest20 = sorted(idf_items, key=lambda x: x[1])[::-1][:20]

# 分離詞彙名稱和 IDF 值
sorted_feature_names_lowest20, sorted_idf_values_lowest20 = zip(*idf_items_sorted_lowest20)
sorted_feature_names_highest20, sorted_idf_values_highest20 = zip(*idf_items_sorted_highest20)
# 繪製 IDF 值最低的部分
plt.figure(figsize=(10, 6))
plt.bar(sorted_feature_names_lowest20, sorted_idf_values_lowest20, color='blue')
plt.xlabel('Words')
plt.ylabel('IDF Values')
plt.title('Top 20 Words with Lowest IDF Values in Spam_SMS.csv')
plt.xticks(rotation=90)
plt.tight_layout()

plt.show()

plt.figure(figsize=(10, 6))
plt.bar(sorted_feature_names_highest20, sorted_idf_values_highest20, color='blue')
plt.xlabel('Words')
plt.ylabel('IDF Values')
plt.title('Top 20 Words with Highest IDF Values in Spam_SMS.csv')
plt.xticks(rotation=90)
plt.tight_layout()

plt.show()
#########################畫出每個詞彙的TF-IDF值########################
'''

# Define all models and vectorizer

vec = CountVectorizer()

model_DT = DecisionTreeClassifier(class_weight='balanced')
model_XGB = XGBClassifier()
model_MNB = MultinomialNB()
model_LR = LogisticRegression(class_weight='balanced')
model_SVC = SVC(class_weight='balanced')
model_RF = RandomForestClassifier(class_weight='balanced')

# Fit the vectorizer and transform the data
X = vec.fit_transform(data['Message'])
y = data['Class'].values

from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

skf_results_normal = []
with Bar('Processing [None]...', max = 5) as bar:
    for train_index, test_index in skf.split(X, y):
        
        accuracy = [0,0,0,0,0,0]
        
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        # Train each model on this fold
        model_MNB.fit(X_train_fold, y_train_fold)
        model_LR.fit(X_train_fold, y_train_fold)
        model_SVC.fit(X_train_fold, y_train_fold)
        model_RF.fit(X_train_fold, y_train_fold)
        model_DT.fit(X_train_fold, y_train_fold)
        model_XGB.fit(X_train_fold, y_train_fold)
        
        pred_MNB = model_MNB.predict(X_test_fold)
        pred_LR = model_LR.predict(X_test_fold)
        pred_SVC = model_SVC.predict(X_test_fold)
        pred_RF = model_RF.predict(X_test_fold)
        pred_DT = model_DT.predict(X_test_fold)
        pred_XGB = model_XGB.predict(X_test_fold)
        
        # Evaluate each model on the test fold
        accuracy[0] = [round(accuracy_score(y_test_fold, pred_MNB)*100, 2), round(precision_score(y_test_fold, pred_MNB)*100, 2), round(recall_score(y_test_fold, pred_MNB)*100, 2), round(f1_score(y_test_fold, pred_MNB)*100, 2)]
        accuracy[1] = [round(accuracy_score(y_test_fold, pred_LR)*100, 2), round(precision_score(y_test_fold, pred_LR)*100, 2), round(recall_score(y_test_fold, pred_LR)*100, 2), round(f1_score(y_test_fold, pred_LR)*100, 2)]
        accuracy[2] = [round(accuracy_score(y_test_fold, pred_SVC)*100, 2), round(precision_score(y_test_fold, pred_SVC)*100, 2), round(recall_score(y_test_fold, pred_SVC)*100, 2), round(f1_score(y_test_fold, pred_SVC)*100, 2)]
        accuracy[3] = [round(accuracy_score(y_test_fold, pred_RF)*100, 2), round(precision_score(y_test_fold, pred_RF)*100, 2), round(recall_score(y_test_fold, pred_RF)*100, 2), round(f1_score(y_test_fold, pred_RF)*100, 2)]
        accuracy[4] = [round(accuracy_score(y_test_fold, pred_DT)*100, 2), round(precision_score(y_test_fold, pred_DT)*100, 2), round(recall_score(y_test_fold, pred_DT)*100, 2), round(f1_score(y_test_fold, pred_DT)*100, 2)]
        accuracy[5] = [round(accuracy_score(y_test_fold, pred_XGB)*100, 2), round(precision_score(y_test_fold, pred_XGB)*100, 2), round(recall_score(y_test_fold, pred_XGB)*100, 2), round(f1_score(y_test_fold, pred_XGB)*100, 2)]
        skf_results_normal.append(accuracy)
        bar.next()
    
print("Fold accuracy [No other preprocessing]:\n", pd.DataFrame(skf_results_normal, columns = ["MNB", "LR", "SVC", "RF", "DT", "XGB"]))
best_results_df = pd.DataFrame(np.max(np.array(skf_results_normal), axis=0), columns=["Accuracy", "Precision", "Recall", "F1-Score"], index=["MNB", "LR", "SVC", "RF", "DT", "XGB"])
print("Best accuracy, precision, recall, and F1-score for each model [No other preprocessing]:\n", best_results_df)

skf_results_smote = []
with Bar('Processing [with SMOTE]...', max = 5) as bar:
    for train_index, test_index in skf.split(X, y):
        
        smote = SMOTE()
        accuracy = [0,0,0,0,0,0]
        
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        
        X_train_fold, y_train_fold = smote.fit_resample(X_train_fold, y_train_fold)

        # Train each model on this fold
        model_MNB.fit(X_train_fold, y_train_fold)
        model_LR.fit(X_train_fold, y_train_fold)
        model_SVC.fit(X_train_fold, y_train_fold)
        model_RF.fit(X_train_fold, y_train_fold)
        model_DT.fit(X_train_fold, y_train_fold)
        model_XGB.fit(X_train_fold, y_train_fold)

        pred_MNB = model_MNB.predict(X_test_fold)
        pred_LR = model_LR.predict(X_test_fold)
        pred_SVC = model_SVC.predict(X_test_fold)
        pred_RF = model_RF.predict(X_test_fold)
        pred_DT = model_DT.predict(X_test_fold)
        pred_XGB = model_XGB.predict(X_test_fold)
        
        # Evaluate each model on the test fold
        accuracy[0] = [round(accuracy_score(y_test_fold, pred_MNB)*100, 2), round(precision_score(y_test_fold, pred_MNB)*100, 2), round(recall_score(y_test_fold, pred_MNB)*100, 2), round(f1_score(y_test_fold, pred_MNB)*100, 2)]
        accuracy[1] = [round(accuracy_score(y_test_fold, pred_LR)*100, 2), round(precision_score(y_test_fold, pred_LR)*100, 2), round(recall_score(y_test_fold, pred_LR)*100, 2), round(f1_score(y_test_fold, pred_LR)*100, 2)]
        accuracy[2] = [round(accuracy_score(y_test_fold, pred_SVC)*100, 2), round(precision_score(y_test_fold, pred_SVC)*100, 2), round(recall_score(y_test_fold, pred_SVC)*100, 2), round(f1_score(y_test_fold, pred_SVC)*100, 2)]
        accuracy[3] = [round(accuracy_score(y_test_fold, pred_RF)*100, 2), round(precision_score(y_test_fold, pred_RF)*100, 2), round(recall_score(y_test_fold, pred_RF)*100, 2), round(f1_score(y_test_fold, pred_RF)*100, 2)]
        accuracy[4] = [round(accuracy_score(y_test_fold, pred_DT)*100, 2), round(precision_score(y_test_fold, pred_DT)*100, 2), round(recall_score(y_test_fold, pred_DT)*100, 2), round(f1_score(y_test_fold, pred_DT)*100, 2)]
        accuracy[5] = [round(accuracy_score(y_test_fold, pred_XGB)*100, 2), round(precision_score(y_test_fold, pred_XGB)*100, 2), round(recall_score(y_test_fold, pred_XGB)*100, 2), round(f1_score(y_test_fold, pred_XGB)*100, 2)]
        
        skf_results_smote.append(accuracy)
        bar.next()
    
print("Fold accuracy [SMOTE]:\n", pd.DataFrame(skf_results_smote, columns = ["MNB", "LR", "SVC", "RF", "DT", "XGB"]))
best_results_df_smote = pd.DataFrame(np.max(np.array(skf_results_smote), axis=0), columns=["Accuracy", "Precision", "Recall", "F1-Score"], index=["MNB", "LR", "SVC", "RF", "DT", "XGB"])
print("Best accuracy, precision, recall, and F1-score for each model [SMOTE]:\n", best_results_df_smote)

skf_results_rus = []
with Bar('Processing [with RandomUnderSampler]...', max = 5) as bar:
    for train_index, test_index in skf.split(X, y):
        
        rus = RandomUnderSampler()
        accuracy = [0,0,0,0,0,0]
        
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        
        X_train_fold, y_train_fold = rus.fit_resample(X_train_fold, y_train_fold)

        # Train each model on this fold
        model_MNB.fit(X_train_fold, y_train_fold)
        model_LR.fit(X_train_fold, y_train_fold)
        model_SVC.fit(X_train_fold, y_train_fold)
        model_RF.fit(X_train_fold, y_train_fold)
        model_DT.fit(X_train_fold, y_train_fold)
        model_XGB.fit(X_train_fold, y_train_fold)

        pred_MNB = model_MNB.predict(X_test_fold)
        pred_LR = model_LR.predict(X_test_fold)
        pred_SVC = model_SVC.predict(X_test_fold)
        pred_RF = model_RF.predict(X_test_fold)
        pred_DT = model_DT.predict(X_test_fold)
        pred_XGB = model_XGB.predict(X_test_fold)
        
        # Evaluate each model on the test fold
        accuracy[0] = [round(accuracy_score(y_test_fold, pred_MNB)*100, 2), round(precision_score(y_test_fold, pred_MNB)*100, 2), round(recall_score(y_test_fold, pred_MNB)*100, 2), round(f1_score(y_test_fold, pred_MNB)*100, 2)]
        accuracy[1] = [round(accuracy_score(y_test_fold, pred_LR)*100, 2), round(precision_score(y_test_fold, pred_LR)*100, 2), round(recall_score(y_test_fold, pred_LR)*100, 2), round(f1_score(y_test_fold, pred_LR)*100, 2)]
        accuracy[2] = [round(accuracy_score(y_test_fold, pred_SVC)*100, 2), round(precision_score(y_test_fold, pred_SVC)*100, 2), round(recall_score(y_test_fold, pred_SVC)*100, 2), round(f1_score(y_test_fold, pred_SVC)*100, 2)]
        accuracy[3] = [round(accuracy_score(y_test_fold, pred_RF)*100, 2), round(precision_score(y_test_fold, pred_RF)*100, 2), round(recall_score(y_test_fold, pred_RF)*100, 2), round(f1_score(y_test_fold, pred_RF)*100, 2)]
        accuracy[4] = [round(accuracy_score(y_test_fold, pred_DT)*100, 2), round(precision_score(y_test_fold, pred_DT)*100, 2), round(recall_score(y_test_fold, pred_DT)*100, 2), round(f1_score(y_test_fold, pred_DT)*100, 2)]
        accuracy[5] = [round(accuracy_score(y_test_fold, pred_XGB)*100, 2), round(precision_score(y_test_fold, pred_XGB)*100, 2), round(recall_score(y_test_fold, pred_XGB)*100, 2), round(f1_score(y_test_fold, pred_XGB)*100, 2)]
        
        skf_results_rus.append(accuracy)
        bar.next()
    
print("Fold accuracy [RUS]:\n", pd.DataFrame(skf_results_rus, columns = ["MNB", "LR", "SVC", "RF", "DT", "XGB"]))
best_results_df_rus = pd.DataFrame(np.max(np.array(skf_results_rus), axis=0), columns=["Accuracy", "Precision", "Recall", "F1-Score"], index=["MNB", "LR", "SVC", "RF", "DT", "XGB"])
print("Best accuracy, precision, recall, and F1-score for each model [RUS]:\n", best_results_df_rus)

for i in range(len(y_test_fold)):
    if y_test_fold[i] == pred_MNB[i]:
        continue
    else:
        if y_test_fold[i] == 0 and pred_MNB[i] == 1:
            print(f'False Negative: {data.iloc[i]["Message"]}')
        else:
            print(f'False Positive: {data.iloc[i]["Message"]}')

'''model_MNB.fit(X_train, y_train)
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
cm_MNB = confusion_matrix(y_test_fold, y_pred_MNB)
cm_LR = confusion_matrix(y_test_fold, y_pred_LR)
cm_SVC = confusion_matrix(y_test_fold, y_pred_SVC)
cm_RF = confusion_matrix(y_test_fold, y_pred_RF)
cm_DT = confusion_matrix(y_test_fold, y_pred_DT)
cm_XGB = confusion_matrix(y_test_fold, y_pred_XGB)'''

'''print("--- Confusion Matrices ---")
print(f"Multinomial Naive Bayes: \n{cm_MNB}")
print(f"Logistic Regression: \n{cm_LR}")
print(f"Support Vector Machine Classification: \n{cm_SVC}")
print(f"Random Forest Classification: \n{cm_RF}")
print(f"Decision Tree Classification: \n{cm_DT}")
print(f"XGBoost Classification: \n{cm_XGB}")'''

def extract_spam_keywords(msg_vector, vectorizer, model):
    feature_names = vectorizer.get_feature_names_out()
    msg_features = msg_vector.toarray()[0]
    spam_keywords = [(feature_names[i], np.exp(model.feature_log_prob_[1][i]) * 100) for i, value in enumerate(msg_features)
                     if value > 0 and model.feature_log_prob_[1][i] > model.feature_log_prob_[0][i]]
    return spam_keywords

fig, axes = plt.subplots(2, 3, figsize=(15, 15))
axes = axes.flatten()

#confusion_matrices = [cm_MNB, cm_LR, cm_SVC, cm_RF, cm_DT, cm_XGB]
titles = ["Multinomial Naive Bayes", "Logistic Regression", "Support Vector Machine", "Random Forest", "Decision Tree", "XGBoost"]

'''for ax, cm, title in zip(axes, confusion_matrices, titles):
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues', cbar=False)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()'''

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
