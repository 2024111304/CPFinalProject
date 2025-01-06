import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from textblob import TextBlob
from progress.bar import Bar
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import string
import re

# 讀取訓練資料集
data = pd.read_csv('Spam_SMS.csv')

# 讀取測試資料集
data_gpt = pd.read_csv('Spam_SMS_ChatGPT_v2.csv')

# Drop duplicates and missing values, then describe the data
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)
data.reset_index(drop=True, inplace=True)

data_gpt.drop_duplicates(inplace=True)
data_gpt.dropna(inplace=True)
data_gpt.reset_index(drop=True, inplace=True)

# Remove punctuation from both data and data_gpt
data['Message'] = data['Message'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
data_gpt['Message'] = data_gpt['Message'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

replace_patterns = [
    (r"can\'t", "cannot"),
    (r"won't", "will not"),
    (r"i'm", "i am"),
    (r"isn't", "is not"),
    (r"(\w+)'ll", "\g<1> will"),
    (r"(\w+)n't", "\g<1> not"),
    (r"(\w+)'ve", "\g<1> have"),
    (r"(\w+)'s", "\g<1> is"),
    (r"(\w+)'re", "\g<1> are"),
    (r"(\w+)'d", "\g<1> would")
]

def replace_abbreviations(text, patterns):
    for pattern, repl in patterns:
        text = re.sub(pattern, repl, text)
    return text

def replace_repeats(text):
    repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')
    repl = r'\1\2\3'
    if text in stopwords.words('english'):
        return text
    repl_text = repeat_regexp.sub(repl, text)
    if repl_text != text:
        return replace_repeats(repl_text)
    else:
        return text

data['Message'] = data['Message'].apply(lambda x: replace_abbreviations(x, replace_patterns))
data_gpt['Message'] = data_gpt['Message'].apply(lambda x: replace_abbreviations(x, replace_patterns))
data['Message'] = data['Message'].apply(lambda x: replace_repeats(x))
data_gpt['Message'] = data_gpt['Message'].apply(lambda x: replace_repeats(x))

#stopwords.word刪除
tokens = [word_tokenize(i) for i in data['Message']]
tokens_gpt = [word_tokenize(i) for i in data_gpt['Message']]
stopwords_list = stopwords.words('english')

processed_tokens = []
for token_list in tokens:
    filtered_tokens = [word for word in token_list if word.lower() not in stopwords_list]
    if filtered_tokens:  # 確保不會移除所有的詞彙
        processed_tokens.append(' '.join(filtered_tokens))
    else:
        processed_tokens.append('dummy')  # 如果全是停用詞，保留一個佔位符詞彙

processed_tokens_gpt = []
for token_list in tokens_gpt:
    filtered_tokens = [word for word in token_list if word.lower() not in stopwords_list]
    if filtered_tokens:  # 確保不會移除所有的詞彙
        processed_tokens_gpt.append(' '.join(filtered_tokens))
    else:
        processed_tokens_gpt.append('dummy')  # 如果全是停用詞，保留一個佔位符詞彙

data['Message'] = processed_tokens
data_gpt['Message'] = processed_tokens_gpt

# 確保所有值都是字串
data['Message'] = data['Message'].astype(str).str.lower()
data_gpt['Message'] = data_gpt['Message'].astype(str).str.lower()
data['Class'] = data['Class'].map({'ham': 0, 'spam': 1})
data_gpt['Class'] = data_gpt['Class'].map({'ham': 0, 'spam': 1})

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['Message'])
y = data['Class'].values

X_gpt = vectorizer.transform(data_gpt['Message'])
y_gpt = data_gpt['Class'].values

#########################畫出每個詞彙的TF-IDF值########################
documents = data['Message'].values

# 初始化 TfidfVectorizer 並擬合數據
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

# Define all models and vectorizer

model_DT = DecisionTreeClassifier(class_weight='balanced')
model_XGB = XGBClassifier()
model_MNB = MultinomialNB()
model_LR = LogisticRegression(class_weight='balanced')
model_SVC = SVC(class_weight='balanced')
model_RF = RandomForestClassifier(class_weight='balanced')

from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

skf_results_normal = []
conf_matrix_normal = []
with Bar('Processing [None]...', max = 5) as bar:
    for train_index, test_index in skf.split(X, y):
        
        accuracy = [0,0,0,0,0,0]
        conf_matrix = []
        
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
        
        conf_matrix.append(confusion_matrix(y_test_fold, pred_MNB))
        conf_matrix.append(confusion_matrix(y_test_fold, pred_LR))
        conf_matrix.append(confusion_matrix(y_test_fold, pred_SVC))
        conf_matrix.append(confusion_matrix(y_test_fold, pred_RF))
        conf_matrix.append(confusion_matrix(y_test_fold, pred_DT))
        conf_matrix.append(confusion_matrix(y_test_fold, pred_XGB))
        conf_matrix_normal.append(conf_matrix)
        
        bar.next()
        
#print("Fold accuracy [No other preprocessing]:\n", pd.DataFrame(skf_results_normal, columns = ["MNB", "LR", "SVC", "RF", "DT", "XGB"]))
best_results_df = pd.DataFrame(np.max(np.array(skf_results_normal), axis=0), columns=["Accuracy", "Precision", "Recall", "F1-Score"], index=["MNB", "LR", "SVC", "RF", "DT", "XGB"])
print("Best accuracy, precision, recall, and F1-score for each model [No other preprocessing]:\n", best_results_df)
conf_matrix_normal = np.array(conf_matrix_normal)

best_acc_instance = [np.argmax(np.array(skf_results_normal)[:,0,0]), np.argmax(np.array(skf_results_normal)[:,1,0]), \
                    np.argmax(np.array(skf_results_normal)[:,2,0]), np.argmax(np.array(skf_results_normal)[:,3,0]), \
                    np.argmax(np.array(skf_results_normal)[:,4,0]), np.argmax(np.array(skf_results_normal)[:,5,0])] # Index of instance with best accuracy for each model

'''print("Conf. Matrix of instance with best accuracy [No other preprocessing]:\nMNB:\n", conf_matrix_normal[best_acc_instance[0], 0], "\nLR:\n", conf_matrix_normal[best_acc_instance[1], 1], \
        "\nSVC:\n", conf_matrix_normal[best_acc_instance[2], 2], "\nRF:\n", conf_matrix_normal[best_acc_instance[3], 3], "\nDT:\n", conf_matrix_normal[best_acc_instance[4], 4], \
        "\nXGB:\n", conf_matrix_normal[best_acc_instance[5], 5])'''

fig, axes = plt.subplots(2, 3, figsize=(15, 15))
axes = axes.flatten()

titles = ["Multinomial Naive Bayes", "Logistic Regression", "Support Vector Machine", "Random Forest", "Decision Tree", "XGBoost"]

confusion_matrices = [conf_matrix_normal[best_acc_instance[0], 0], conf_matrix_normal[best_acc_instance[1], 1], conf_matrix_normal[best_acc_instance[2], 2], conf_matrix_normal[best_acc_instance[3], 3], conf_matrix_normal[best_acc_instance[4], 4], conf_matrix_normal[best_acc_instance[5], 5]]

for ax, cm, title in zip(axes, confusion_matrices, titles):
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues', cbar=False, square=True)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()

skf_results_smote = []
conf_matrix_smote = []
with Bar('Processing [with SMOTE]...', max = 5) as bar:
    for train_index, test_index in skf.split(X, y):
        
        smote = SMOTE(sampling_strategy=0.7, random_state=42)
        accuracy = [0,0,0,0,0,0]
        conf_matrix = []
        
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
        
        conf_matrix.append(confusion_matrix(y_test_fold, pred_MNB))
        conf_matrix.append(confusion_matrix(y_test_fold, pred_LR))
        conf_matrix.append(confusion_matrix(y_test_fold, pred_SVC))
        conf_matrix.append(confusion_matrix(y_test_fold, pred_RF))
        conf_matrix.append(confusion_matrix(y_test_fold, pred_DT))
        conf_matrix.append(confusion_matrix(y_test_fold, pred_XGB))
        conf_matrix_smote.append(conf_matrix)
        
        bar.next()
conf_matrix_smote = np.array(conf_matrix_smote)
#print("Fold accuracy [SMOTE]:\n", pd.DataFrame(skf_results_smote, columns = ["MNB", "LR", "SVC", "RF", "DT", "XGB"]))
best_results_df_smote = pd.DataFrame(np.max(np.array(skf_results_smote), axis=0), columns=["Accuracy", "Precision", "Recall", "F1-Score"], index=["MNB", "LR", "SVC", "RF", "DT", "XGB"])
print("Best accuracy, precision, recall, and F1-score for each model [SMOTE] (%):\n", best_results_df_smote)
conf_matrix_smote = np.array(conf_matrix_smote)

best_acc_instance = [np.argmax(np.array(skf_results_smote)[:,0,0]), np.argmax(np.array(skf_results_smote)[:,1,0]), \
                    np.argmax(np.array(skf_results_smote)[:,2,0]), np.argmax(np.array(skf_results_smote)[:,3,0]), \
                    np.argmax(np.array(skf_results_smote)[:,4,0]), np.argmax(np.array(skf_results_smote)[:,5,0])] # Index of instance with best accuracy for each model 
'''print("Conf. Matrix of instance with best accuracy [SMOTE]:\nMNB:\n", conf_matrix_smote[best_acc_instance[0], 0], "\nLR:\n", conf_matrix_smote[best_acc_instance[1], 1], \
        "\nSVC:\n", conf_matrix_smote[best_acc_instance[2], 2], "\nRF:\n", conf_matrix_smote[best_acc_instance[3], 3], "\nDT:\n", conf_matrix_smote[best_acc_instance[4], 4], \
        "\nXGB:\n", conf_matrix_smote[best_acc_instance[5], 5])'''

fig, axes = plt.subplots(2, 3, figsize=(15, 15))
axes = axes.flatten()

titles = ["Multinomial Naive Bayes", "Logistic Regression", "Support Vector Machine", "Random Forest", "Decision Tree", "XGBoost"]

confusion_matrices = [conf_matrix_smote[best_acc_instance[0], 0], conf_matrix_smote[best_acc_instance[1], 1], conf_matrix_smote[best_acc_instance[2], 2], conf_matrix_smote[best_acc_instance[3], 3], conf_matrix_smote[best_acc_instance[4], 4], conf_matrix_smote[best_acc_instance[5], 5]]

for ax, cm, title in zip(axes, confusion_matrices, titles):
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Oranges', cbar=False, square=True)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()

skf_results_rus = []
conf_matrix_rus = []
with Bar('Processing [with RandomUnderSampler]...', max = 5) as bar:
    for train_index, test_index in skf.split(X, y):
        
        rus = RandomUnderSampler(sampling_strategy=0.7, random_state=42)
        accuracy = [0,0,0,0,0,0]
        conf_matrix = []
        
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
        conf_matrix.append(confusion_matrix(y_test_fold, pred_MNB))
        conf_matrix.append(confusion_matrix(y_test_fold, pred_LR))
        conf_matrix.append(confusion_matrix(y_test_fold, pred_SVC))
        conf_matrix.append(confusion_matrix(y_test_fold, pred_RF))
        conf_matrix.append(confusion_matrix(y_test_fold, pred_DT))
        conf_matrix.append(confusion_matrix(y_test_fold, pred_XGB))
        conf_matrix_rus.append(conf_matrix)
        
        bar.next()
conf_matrix_rus = np.array(conf_matrix_rus)
#print("Fold accuracy [RUS]:\n", pd.DataFrame(skf_results_rus, columns = ["MNB", "LR", "SVC", "RF", "DT", "XGB"]))
best_results_df_rus = pd.DataFrame(np.max(np.array(skf_results_rus), axis=0), columns=["Accuracy", "Precision", "Recall", "F1-Score"], index=["MNB", "LR", "SVC", "RF", "DT", "XGB"])
print("Best accuracy, precision, recall, and F1-score for each model [RUS] (%):\n", best_results_df_rus)
conf_matrix_rus = np.array(conf_matrix_rus)

best_acc_instance = [np.argmax(np.array(skf_results_rus)[:,0,0]), np.argmax(np.array(skf_results_rus)[:,1,0]), \
                    np.argmax(np.array(skf_results_rus)[:,2,0]), np.argmax(np.array(skf_results_rus)[:,3,0]), \
                    np.argmax(np.array(skf_results_rus)[:,4,0]), np.argmax(np.array(skf_results_rus)[:,5,0])] # Index of instance with best accuracy for each model
'''print("Conf. Matrix of instance with best accuracy [RUS]:\nMNB:\n", conf_matrix_rus[best_acc_instance[0], 0], "\nLR:\n", conf_matrix_rus[best_acc_instance[1], 1], \
        "\nSVC:\n", conf_matrix_rus[best_acc_instance[2], 2], "\nRF:\n", conf_matrix_rus[best_acc_instance[3], 3], "\nDT:\n", conf_matrix_rus[best_acc_instance[4], 4], \
        "\nXGB:\n", conf_matrix_rus[best_acc_instance[5], 5])'''

fig, axes = plt.subplots(2, 3, figsize=(15, 15))
axes = axes.flatten()

titles = ["Multinomial Naive Bayes", "Logistic Regression", "Support Vector Machine", "Random Forest", "Decision Tree", "XGBoost"]

confusion_matrices = [conf_matrix_rus[best_acc_instance[0], 0], conf_matrix_rus[best_acc_instance[1], 1], conf_matrix_rus[best_acc_instance[2], 2], conf_matrix_rus[best_acc_instance[3], 3], conf_matrix_rus[best_acc_instance[4], 4], conf_matrix_rus[best_acc_instance[5], 5]]

for ax, cm, title in zip(axes, confusion_matrices, titles):
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Greens', cbar=False, square=True)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()

# Plotting the multi-bar chart for best accuracy, precision, recall, F1-score for each model

metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
models = ["MNB", "LR", "SVC", "RF", "DT", "XGB"]

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for i, metric in enumerate(metrics):
    ax = axes[i]
    index = np.arange(len(models))
    bar_width = 0.2

    no_processing = best_results_df[metric].values
    smote = best_results_df_smote[metric].values
    rus = best_results_df_rus[metric].values

    ax.bar(index, no_processing, bar_width, label='No Processing')
    ax.bar(index + bar_width, smote, bar_width, label='SMOTE')
    ax.bar(index + 2 * bar_width, rus, bar_width, label='RUS')

    ax.set_xlabel('Models')
    ax.set_ylabel(metric)
    ax.set_ylim(0, 100)
    ax.set_title(f'Best {metric} for Each Model')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(models)
    #ax.legend()

plt.tight_layout()
plt.show()

'''for i in range(len(y_test_fold)):
    if y_test_fold[i] == pred_MNB[i]:
        continue
    else:
        if y_test_fold[i] == 0 and pred_MNB[i] == 1:
            print(f'False Negative: {data.iloc[i]["Message"]}')
        else:
            print(f'False Positive: {data.iloc[i]["Message"]}')'''

def extract_spam_keywords(msg_vector, vectorizer, model):
    feature_names = vectorizer.get_feature_names_out()
    msg_features = msg_vector.toarray()[0]
    spam_keywords = [(feature_names[i], np.exp(model.feature_log_prob_[1][i]) * 100) for i, value in enumerate(msg_features)
                     if value > 0 and model.feature_log_prob_[1][i] > model.feature_log_prob_[0][i]]
    return spam_keywords

y_pred_gpt_MNB = model_MNB.predict(X_gpt)
y_pred_gpt_LR = model_LR.predict(X_gpt)
y_pred_gpt_SVC = model_SVC.predict(X_gpt)
y_pred_gpt_RF = model_RF.predict(X_gpt)
y_pred_gpt_DT = model_DT.predict(X_gpt)
y_pred_gpt_XGB = model_XGB.predict(X_gpt)

print(f"MNB (GPT): ACC. {accuracy_score(y_gpt, y_pred_gpt_MNB)*100:.2f}%, PRC. {precision_score(y_gpt, y_pred_gpt_MNB)*100:.2f}%, RCL. {recall_score(y_gpt, y_pred_gpt_MNB)*100:.2f}%, F1. {f1_score(y_gpt, y_pred_gpt_MNB)*100:.2f}%")
print(f"LR  (GPT): ACC. {accuracy_score(y_gpt, y_pred_gpt_LR)*100:.2f}%, PRC. {precision_score(y_gpt, y_pred_gpt_LR)*100:.2f}%, RCL. {recall_score(y_gpt, y_pred_gpt_LR)*100:.2f}%, F1. {f1_score(y_gpt, y_pred_gpt_LR)*100:.2f}%")
print(f"SVC (GPT): ACC. {accuracy_score(y_gpt, y_pred_gpt_SVC)*100:.2f}%, PRC. {precision_score(y_gpt, y_pred_gpt_SVC)*100:.2f}%, RCL. {recall_score(y_gpt, y_pred_gpt_SVC)*100:.2f}%, F1. {f1_score(y_gpt, y_pred_gpt_SVC)*100:.2f}%")
print(f"RF  (GPT): ACC. {accuracy_score(y_gpt, y_pred_gpt_RF)*100:.2f}%, PRC. {precision_score(y_gpt, y_pred_gpt_RF)*100:.2f}%, RCL. {recall_score(y_gpt, y_pred_gpt_RF)*100:.2f}%, F1. {f1_score(y_gpt, y_pred_gpt_RF)*100:.2f}%")
print(f"DT  (GPT): ACC. {accuracy_score(y_gpt, y_pred_gpt_DT)*100:.2f}%, PRC. {precision_score(y_gpt, y_pred_gpt_DT)*100:.2f}%, RCL. {recall_score(y_gpt, y_pred_gpt_DT)*100:.2f}%, F1. {f1_score(y_gpt, y_pred_gpt_DT)*100:.2f}%")
print(f"XGB (GPT): ACC. {accuracy_score(y_gpt, y_pred_gpt_XGB)*100:.2f}%, PRC. {precision_score(y_gpt, y_pred_gpt_XGB)*100:.2f}%, RCL. {recall_score(y_gpt, y_pred_gpt_XGB)*100:.2f}%, F1. {f1_score(y_gpt, y_pred_gpt_XGB)*100:.2f}%")

while True:
    msg = input("Enter testing message (enter nothing to quit): ").lower()
    if not msg:
        break
    msg = vectorizer.transform([msg])
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
        keywords_list = extract_spam_keywords(msg, vectorizer, model_MNB)
        
        # Tally the percentages
        keyword_dict = {}
        for keyword, prob in keywords_list:
            if keyword in keyword_dict:
                keyword_dict[keyword].append(prob)
            else:
                keyword_dict[keyword] = [prob]
        
        # Calculate the average percentage
        averaged_keywords = [(keyword, np.mean(probs)) for keyword, probs in keyword_dict.items()]
        
        # Sort by the average percentage in descending order
        averaged_keywords_sorted = sorted(averaged_keywords, key=lambda x: x[1], reverse=True)
        
        print(f"Keywords indicating spam: ")
        for keyword, avg_prob in averaged_keywords_sorted:
            print(f"{keyword} (Avg. Prob.: {avg_prob:.3f}%)")
    else:
        print("This message is NOT flagged as spam.")
