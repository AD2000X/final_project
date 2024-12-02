# Basic libraries
import numpy as np
import pandas as pd
import json
import re
from collections import Counter
import gc

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, roc_auc_score,
    make_scorer, f1_score, auc, precision_recall_curve
)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE

# Natural Language Processing
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.util import ngrams
import spacy
import textstat
from wordcloud import WordCloud

# Statistical analysis
from scipy import stats

# File and system operations
import os
import glob
from google.colab import drive
import joblib

# Time measurement
from time import time

# NLTK downloads
nltk.download('vader_lexicon')

# spaCy model loading
nlp = spacy.load("en_core_web_sm")

# Installation commands (these should be run separately, not imported)
# !pip install catboost textstat nltk pandas scikit-learn
# !python -m spacy download en_core_web_sm

# Jupyter magic command (should be used in Jupyter notebooks)
# Basic libraries
import numpy as np
import pandas as pd
import json
import re
from collections import Counter
import gc

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, roc_auc_score,
    make_scorer, f1_score, auc, precision_recall_curve
)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE

# Natural Language Processing
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.util import ngrams
import spacy
import textstat
from wordcloud import WordCloud

# Statistical analysis
from scipy import stats

# File and system operations
import os
import glob
from google.colab import drive
import joblib

# Time measurement
from time import time

# NLTK downloads
nltk.download('vader_lexicon')

# spaCy model loading
nlp = spacy.load("en_core_web_sm")

# Installation commands (these should be run separately, not imported)
# !pip install catboost textstat nltk pandas scikit-learn
# !python -m spacy download en_core_web_sm

# Jupyter magic command (should be used in Jupyter notebooks)
# %%time

# mount Google drive
drive.mount('/content/drive', force_remount=True)
base_dir = '/content/drive/My Drive/seperate_0731'
file_path = f'{base_dir}/MIRUKU.csv'

"""# EDA"""

# reading the CSV file using different encodings
encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'utf-16']
for encoding in encodings:
    try:
        df_original = pd.read_csv(file_path, encoding=encoding)
        print(f'Successfully read the file with encoding: {encoding}')
        print(df_original.head())
        break
    except UnicodeDecodeError as e:
        print(f'Failed to read the file with encoding: {encoding}')
        print(f'Error: {e}')

df_original.head()

# examine data distribution
stratify_col = 'sensation'

print("\nOriginal class distribution:")
print(df_original[stratify_col].value_counts())

print(df_original.shape)
print(df_original.columns)
print(df_original.dtypes)

"""## Length of headline"""

# calculate title length and plot histogram
title_length = df_original['headline'].str.split().str.len()

print(title_length.describe())

plt.hist(title_length, bins=20)
plt.title('Distribution of Title Lengths')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.show()

"""## Keyword extraction

### with stopword
"""

# using TF-IDF to identify important keywords
vectorizer = TfidfVectorizer(max_features=100)
tfidf_matrix = vectorizer.fit_transform(df_original['headline'])
feature_names = vectorizer.get_feature_names_out()
print("Top keywords:", feature_names)

# create a dictionary of words and their TF-IDF scores
tfidf_scores = dict(zip(feature_names, tfidf_matrix.sum(axis=0).tolist()[0]))

wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(tfidf_scores)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Top Keywords in Headlines')
plt.show()

"""### w/o stopword"""

# load spaCy English model
nlp = spacy.load("en_core_web_sm")

# function to get spaCy stopwords
def get_spacy_stop_words():
    return spacy.lang.en.stop_words.STOP_WORDS

# get spaCy stopwords
spacy_stop_words = get_spacy_stop_words()

# create a custom TF-IDF Vectorizer that removes stopwords
vectorizer = TfidfVectorizer(max_features=100, stop_words=list(spacy_stop_words))
tfidf_matrix = vectorizer.fit_transform(df_original['headline'])

# get the feature names (i.e., the top keywords)
feature_names = vectorizer.get_feature_names_out()
print("Top keywords:", feature_names)

# create a dictionary of words and their TF-IDF scores
tfidf_scores = dict(zip(feature_names, tfidf_matrix.sum(axis=0).tolist()[0]))

# create and generate a word cloud image
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(tfidf_scores)

# display the generated image
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Top Keywords in Headlines (Without Stopwords)')
plt.show()

"""# Threshold

## XGBoost: Calculate threshold by using arousal score and by using PCA on emotions
"""

df_threshold = df_original.copy()

# data preprocessing
df_threshold['arousal'] = df_threshold['arousal'].map({'Yes': 1, 'No': 0})

emotions = ['joy', 'surprise', 'anticipation', 'trust', 'anger', 'fear', 'sadness', 'disgust', 'neutral']
for emotion in emotions:
    df_threshold[emotion] = df_threshold['emotion'].apply(lambda x: 1 if emotion in x else 0)

emotion_data = df_threshold[emotions]

pca = PCA(n_components=1)
principal_component = pca.fit_transform(emotion_data)
df_threshold['emotion_pca'] = principal_component

scaler = MinMaxScaler()
df_threshold[['sensation_score', 'arousal_score', 'emotion_pca']] = scaler.fit_transform(df_threshold[['sensation_score', 'arousal_score', 'emotion_pca']])

# features and target variables
X = df_threshold[['sensation_score', 'arousal_score', 'emotion_pca']]
y = df_threshold['sensation']

# split data
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

# XGBoost
model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# best threshold on validation set
val_predictions = model.predict_proba(X_val)[:, 1]
fpr, tpr, thresholds = roc_curve(y_val, val_predictions)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal Threshold: {optimal_threshold}")

# final evaluation on test set
test_predictions = model.predict_proba(X_test)[:, 1]
test_predictions_binary = (test_predictions >= optimal_threshold).astype(int)

print("\nTest Set Evaluation:")
print(classification_report(y_test, test_predictions_binary))

# ROC curve with best threshold
plt.figure()
plt.plot(fpr, tpr, marker='.')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Validation Set)')

optimal_fpr = fpr[optimal_idx]
optimal_tpr = tpr[optimal_idx]
plt.plot(optimal_fpr, optimal_tpr, 'ro')
plt.annotate(f'Optimal Threshold: {optimal_threshold:.4f}',
             xy=(optimal_fpr, optimal_tpr),
             xytext=(optimal_fpr + 0.1, optimal_tpr - 0.1),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.show()

# AUC of test set
test_auc = roc_auc_score(y_test, test_predictions)
print(f"\nTest Set AUC: {test_auc}")

# apply combined scores on the test set
df_test = pd.DataFrame(X_test, columns=['sensation_score', 'arousal_score', 'emotion_pca'])
df_test['true_sensation'] = y_test
df_test['combined_score'] = test_predictions
df_test['predicted_sensation'] = test_predictions_binary

print("\nTest Set Results (First 10 rows):")
print(df_test.head(10))

# save best XGBoost threshold and model
joblib.dump(optimal_threshold, f'{base_dir}/XGoptimal_threshold.joblib')
model.save_model(f'{base_dir}/xgboost_threshold_model.json')

"""## ADABoost: Calculate threshold by using arousal score and by using PCA on emotions"""

df_threshold = df_original.copy()

# data preprocessing
df_threshold['arousal'] = df_threshold['arousal'].map({'Yes': 1, 'No': 0})

emotions = ['joy', 'surprise', 'anticipation', 'trust', 'anger', 'fear', 'sadness', 'disgust', 'neutral']
for emotion in emotions:
    df_threshold[emotion] = df_threshold['emotion'].apply(lambda x: 1 if emotion in x else 0)

emotion_data = df_threshold[emotions]

pca = PCA(n_components=1)
principal_component = pca.fit_transform(emotion_data)
df_threshold['emotion_pca'] = principal_component

scaler = MinMaxScaler()
df_threshold[['sensation_score', 'arousal_score', 'emotion_pca']] = scaler.fit_transform(df_threshold[['sensation_score', 'arousal_score', 'emotion_pca']])

# features and target variables
X = df_threshold[['sensation_score', 'arousal_score', 'emotion_pca']]
y = df_threshold['sensation']

# data split
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

# Adaboost
base_estimator = DecisionTreeClassifier(max_depth=1)
model = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# best threshold on validation set
val_predictions = model.predict_proba(X_val)[:, 1]
fpr, tpr, thresholds = roc_curve(y_val, val_predictions)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal Threshold: {optimal_threshold}")

# final evaluation on test set
test_predictions = model.predict_proba(X_test)[:, 1]
test_predictions_binary = (test_predictions >= optimal_threshold).astype(int)

print("\nTest Set Evaluation:")
print(classification_report(y_test, test_predictions_binary))

# ROC curve with best threshold
plt.figure()
plt.plot(fpr, tpr, marker='.')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Validation Set)')

optimal_fpr, optimal_tpr = fpr[optimal_idx], tpr[optimal_idx]
plt.plot(optimal_fpr, optimal_tpr, 'ro')
plt.annotate(f'Optimal Threshold: {optimal_threshold:.4f}',
             xy=(optimal_fpr, optimal_tpr),
             xytext=(optimal_fpr + 0.1, optimal_tpr - 0.1),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.show()

# AUC of test set
test_auc = roc_auc_score(y_test, test_predictions)
print(f"\nTest Set AUC: {test_auc}")

# apply combined scores on the test set
df_test = pd.DataFrame(X_test, columns=['sensation_score', 'arousal_score', 'emotion_pca'])
df_test['true_sensation'] = y_test
df_test['combined_score'] = test_predictions
df_test['predicted_sensation'] = test_predictions_binary

print("\nTest Set Results (First 10 rows):")
print(df_test.head(10))

# save best XGBoost threshold and model
joblib.dump(optimal_threshold, f'{base_dir}/ADAoptimal_threshold.joblib')
joblib.dump(model, f'{base_dir}/adaboost_model.joblib')

"""## CATBoost: Calculate threshold by using arousal score and by using PCA on emotions"""

df_threshold = df_original.copy()

# data preprocessing
df_threshold['arousal'] = df_threshold['arousal'].map({'Yes': 1, 'No': 0})

emotions = ['joy', 'surprise', 'anticipation', 'trust', 'anger', 'fear', 'sadness', 'disgust', 'neutral']
for emotion in emotions:
    df_threshold[emotion] = df_threshold['emotion'].apply(lambda x: 1 if emotion in x else 0)

emotion_data = df_threshold[emotions]

pca = PCA(n_components=1)
principal_component = pca.fit_transform(emotion_data)
df_threshold['emotion_pca'] = principal_component

scaler = MinMaxScaler()
df_threshold[['sensation_score', 'arousal_score', 'emotion_pca']] = scaler.fit_transform(df_threshold[['sensation_score', 'arousal_score', 'emotion_pca']])

# features and target variables
X = df_threshold[['sensation_score', 'arousal_score', 'emotion_pca']]
y = df_threshold['sensation']

# data split
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

# CatBoost
model = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1, loss_function='Logloss', verbose=False)
model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=20, verbose=False)

# best threshold on validation set
val_predictions = model.predict_proba(X_val)[:, 1]
fpr, tpr, thresholds = roc_curve(y_val, val_predictions)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal Threshold: {optimal_threshold}")

# final evaluation on test set
test_predictions = model.predict_proba(X_test)[:, 1]
test_predictions_binary = (test_predictions >= optimal_threshold).astype(int)

print("\nTest Set Evaluation:")
print(classification_report(y_test, test_predictions_binary))

# ROC curve with best threshold
plt.figure()
plt.plot(fpr, tpr, marker='.')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (Validation Set)')

optimal_fpr, optimal_tpr = fpr[optimal_idx], tpr[optimal_idx]
plt.plot(optimal_fpr, optimal_tpr, 'ro')
plt.annotate(f'Optimal Threshold: {optimal_threshold:.4f}',
             xy=(optimal_fpr, optimal_tpr),
             xytext=(optimal_fpr + 0.1, optimal_tpr - 0.1),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.show()

# AUC of test set
test_auc = roc_auc_score(y_test, test_predictions)
print(f"\nTest Set AUC: {test_auc}")

# apply combined scores on the test set
df_test = pd.DataFrame(X_test, columns=['sensation_score', 'arousal_score', 'emotion_pca'])
df_test['true_sensation'] = y_test
df_test['combined_score'] = test_predictions
df_test['predicted_sensation'] = test_predictions_binary

print("\nTest Set Results (First 10 rows):")
print(df_test.head(10))

# save best CatBoost threshold and model
joblib.dump(optimal_threshold, f'{base_dir}/CAToptimal_threshold.joblib')
model.save_model(f'{base_dir}/catboost_model.cbm')

# print feature importance
feature_importance = model.get_feature_importance()
feature_names = X.columns
for score, name in sorted(zip(feature_importance, feature_names), reverse=True):
    print(f"{name}: {score}")

"""# Features

## Number of words in the headline
"""

# dataframe number of words
df_now = df_original.copy()

# calculate words in headlines
df_now['Number of Words'] = df_now['headline'].apply(lambda x: len(str(x).split()))

print(df_now.head())

min_words = df_now['Number of Words'].min()
max_words = df_now['Number of Words'].max()
avg_words = df_now['Number of Words'].mean()

print(f"min number of words: {min_words}")
print(f"max number of words: {max_words}")
print(f"average words: {avg_words:.2f}")

# calculate the number of stop words for sensation and non-sensation categories, grouped by word count
sensation_counts = df_now[df_now['sensation'] == 1]['Number of Words'].value_counts().sort_index()
non_sensation_counts = df_now[df_now['sensation'] == 0]['Number of Words'].value_counts().sort_index()

# calculate the total number of stop words in each category
total_sensation = sensation_counts.sum()
total_non_sensation = non_sensation_counts.sum()

print("Sensation - Number of Words Counts:")
print(sensation_counts)
print(f"Total Sensation Count: {total_sensation}")

print("\nNon-Sensation - Number of Words Counts:")
print(non_sensation_counts)
print(f"Total Non-Sensation Count: {total_non_sensation}")

# kernel density estimation plot
plt.figure(figsize=(12, 6))
sns.kdeplot(data=df_now, x='Number of Words', hue='sensation', shade=True)
plt.title('Distribution of Number of Words in Headlines', fontsize=16)
plt.xlabel('Number of Words', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(title='Sensation', labels=['Non-Sensation', 'Sensation'])
plt.tight_layout()
plt.show()

# descriptive statistics
desc_stats = df_now.groupby('sensation')['Number of Words'].describe()
print("Descriptive Statistics for Number of Words:")
print(desc_stats)

# perform t-test
sensation_words = df_now[df_now['sensation'] == 1]['Number of Words']
non_sensation_words = df_now[df_now['sensation'] == 0]['Number of Words']
t_stat, p_value = stats.ttest_ind(sensation_words, non_sensation_words)

print(f"\nt-test results for Number of Words:")
print(f"t-statistic: {t_stat}")
print(f"p-value: {p_value}")

# save .csv in utf-8
output_path = f'{base_dir}/df_now.csv'
df_now.to_csv(output_path, index=False, encoding='utf-8')

"""### XGBoost"""

base_dir = '/content/drive/My Drive/seperate_0731'

# # load threshold
# optimal_threshold = joblib.load(f'{base_dir}/XGoptimal_threshold.joblib')

# remove target variables and headlines columns
def prepare_data(df_now):
    X = df_now.drop(['headline', 'clickbait', 'sensation_score',
                   'sensation_reason', 'emotion', 'arousal', 'arousal_score',
                   'arousal_reason', 'sensation'], axis=1)
    y = df_now['sensation']

    # split dataset
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # ensure all data in numeric
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    # preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])

    # fit and transform on train set
    X_train_processed = pipeline.fit_transform(X_train)

    # apply SMOTE on train set
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

    # transform on validation set
    X_val_processed = pipeline.transform(X_val)

    return X_train_resampled, X_val_processed, y_train_resampled, y_val, pipeline

# Prepare data
X_train, X_val, y_train, y_val, preprocessor = prepare_data(df_now)

# Create XGBoost classifier
xgb_classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

param_dist_xgb = {
    'max_depth': [3, 6, 9],
    'min_child_weight': [1, 3],
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1]
}

# Create K-Fold cross-validation object
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

random_search_xg = RandomizedSearchCV(
    estimator=xgb_classifier,
    param_distributions=param_dist_xgb,
    n_iter=100,
    cv=kfold,
    n_jobs=-1,
    verbose=2,
    scoring='f1',
    random_state=42
)

# Fit on preprocessed training data
random_search_xg.fit(X_train, y_train)

print("Best parameters:", random_search_xg.best_params_)
print("Best cross-validation score:", random_search_xg.best_score_)

# Use best parameters to create new model
best_params = random_search_xg.best_params_
best_model = xgb.XGBClassifier(
    **best_params,
    use_label_encoder=False,
    eval_metric='logloss',
    early_stopping_rounds=50  # Add early stopping here
)

# Train the model with early stopping
best_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# Use best model for prediction on validation set
y_val_pred = best_model.predict(X_val)

# Print classification report
print("\nValidation Set Classification Report:")
print(classification_report(y_val, y_val_pred))

"""### ROC AUC"""

# ROC AUC Curve
fpr, tpr, _ = roc_curve(y_val, best_model.predict_proba(X_val)[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_val, best_model.predict_proba(X_val)[:, 1])
pr_auc = auc(recall, precision)

plt.figure(figsize=(10, 8))
plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# !pip install shap

"""## Number of stop words in the headlines"""

# calculate stopwords function
def count_stopwords(text):
    doc = nlp(text)
    return sum(1 for token in doc if token.is_stop)

# copy of Number of stop words in the headlines
df_nostop = df_original.copy()
df_nostop['Number of stop words'] = df_nostop['headline'].apply(count_stopwords)

print(df_nostop.head())

# calculate the number of stop words for sensation and non-sensation categories, grouped by word count
sensation_stopwords_counts = df_nostop[df_nostop['sensation'] == 1]['Number of stop words'].value_counts().sort_index()
non_sensation_stopwords_counts = df_nostop[df_nostop['sensation'] == 0]['Number of stop words'].value_counts().sort_index()

# calculate the total number of stop words in each category
total_sensation_stopwords = sensation_stopwords_counts.sum()
total_non_sensation_stopwords = non_sensation_stopwords_counts.sum()

# print the results grouped by word count
print("Sensation - Number of Stop Words by Word Count:")
print(sensation_stopwords_counts)

print("\nNon-Sensation - Number of Stop Words by Word Count:")
print(non_sensation_stopwords_counts)

# print the total for each category
print(f"\nTotal Stop Words in Sensation Headlines: {total_sensation_stopwords}")
print(f"Total Stop Words in Non-Sensation Headlines: {total_non_sensation_stopwords}")

# Kernel density estimation plot
plt.figure(figsize=(12, 6))
sns.kdeplot(data=df_nostop, x='Number of stop words', hue='sensation', shade=True)
plt.title('Distribution of Number of Stop Words', fontsize=16)
plt.xlabel('Number of Stop Words', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(title='Sensation', labels=['Non-Sensation', 'Sensation'])
plt.tight_layout()
plt.show()

# Descriptive statistics
desc_stats = df_nostop.groupby('sensation')['Number of stop words'].describe()
print("Descriptive Statistics for Number of Stop Words:")
print(desc_stats)

# T test
sensation_stopwords = df_nostop[df_nostop['sensation'] == 1]['Number of stop words']
non_sensation_stopwords = df_nostop[df_nostop['sensation'] == 0]['Number of stop words']

t_stat, p_value = stats.ttest_ind(sensation_stopwords, non_sensation_stopwords)

print("\nT-Test Results for Number of Stop Words:")
print(f"t-statistic: {t_stat}")
print(f"p-value: {p_value}")

# save .csv in utf-8
output_path = f'{base_dir}/df_nostop.csv'
df_nostop.to_csv(output_path, index=False, encoding='utf-8')

"""### XGBoost"""

base_dir = '/content/drive/My Drive/seperate_0731'

# remove target variables and headlines columns
def prepare_data(df_nostop):
    X = df_nostop.drop(['headline', 'clickbait', 'sensation_score',
                   'sensation_reason', 'emotion', 'arousal', 'arousal_score',
                   'arousal_reason', 'sensation'], axis=1)
    y = df_nostop['sensation']

    # # split dataset
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # ensure all data in numeric
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    # preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])

    # fit and transform on train set
    X_train_processed = pipeline.fit_transform(X_train)

    # apply SMOTE on train set
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

    # transform on validation set
    X_val_processed = pipeline.transform(X_val)

    return X_train_resampled, X_val_processed, y_train_resampled, y_val, pipeline

# prepare data
X_train, X_val, y_train, y_val, preprocessor = prepare_data(df_nostop)

# create XGBoost classifier
xgb_classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

param_dist_xgb = {
    'max_depth': [3, 6, 9],
    'min_child_weight': [1, 3],
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1]
}

# create K-Fold cross-validation object
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

random_search_xg = RandomizedSearchCV(
    estimator=xgb_classifier,
    param_distributions=param_dist_xgb,
    n_iter=100,
    cv=kfold,
    n_jobs=-1,
    verbose=2,
    scoring='f1',
    random_state=42
)

# fit on preprocessed training data
random_search_xg.fit(X_train, y_train)

print("Best parameters:", random_search_xg.best_params_)
print("Best cross-validation score:", random_search_xg.best_score_)

# use best parameters to create new model
best_params = random_search_xg.best_params_
best_model = xgb.XGBClassifier(
    **best_params,
    use_label_encoder=False,
    eval_metric='logloss',
    early_stopping_rounds=50  # Add early stopping here
)

# train the model with early stopping
best_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# use best model for prediction on validation set
y_val_pred = best_model.predict(X_val)

# print classification report
print("\nValidation Set Classification Report:")
print(classification_report(y_val, y_val_pred))

"""### ROC AUC"""

# ROC AUC Curve
fpr, tpr, _ = roc_curve(y_val, best_model.predict_proba(X_val)[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_val, best_model.predict_proba(X_val)[:, 1])
pr_auc = auc(recall, precision)

plt.figure(figsize=(10, 8))
plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

"""## The ratio of the number of stop words to the number of content words"""

# dataframe of ratio of the number of stop words to the number of content words
df_ratiostopwords = df_original.copy()

# function of calculating stopwrods and content words
def calculate_stopword_ratio(headline):
    doc = nlp(headline)
    stop_words = [token.text for token in doc if token.is_stop]
    content_words = [token.text for token in doc if not token.is_stop and token.is_alpha]   # consider only alphabetic content words
    if len(content_words) == 0:  # prevent division by zero error
        return 0
    return len(stop_words) / len(content_words)

# calculate the ratio of stop words to content words and store it in a new column
df_ratiostopwords['ratio_stopwords'] = df_ratiostopwords['headline'].apply(calculate_stopword_ratio)

print(df_ratiostopwords[['headline', 'ratio_stopwords']].head())

# kernel density estimation plot
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df_ratiostopwords, x='ratio_stopwords', hue='sensation', shade=True)
plt.title('Distribution of Ratio of Stop Words to Content Words')
plt.xlabel('Ratio of Stop Words to Content Words')
plt.ylabel('Density')
plt.legend(title='Sensation', labels=['Non-Sensation', 'Sensation'])
plt.show()

# descriptive statistics
desc_stats = df_ratiostopwords.groupby('sensation')['ratio_stopwords'].describe()
print("Descriptive Statistics:")
print(desc_stats)

# perform t-test
sensation_ratios = df_ratiostopwords[df_ratiostopwords['sensation'] == 1]['ratio_stopwords']
non_sensation_ratios = df_ratiostopwords[df_ratiostopwords['sensation'] == 0]['ratio_stopwords']
t_stat, p_value = stats.ttest_ind(sensation_ratios, non_sensation_ratios)

print(f"\nt-test results for Ratio of Stop Words to Content Words:")
print(f"t-statistic: {t_stat}")
print(f"p-value: {p_value}")

# save .csv in utf-8
output_path = f'{base_dir}/df_ratiostopwords.csv'
df_ratiostopwords.to_csv(output_path, index=False, encoding='utf-8')

"""### XGBoost"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# base_dir = '/content/drive/My Drive/seperate_0731'
# 
# def prepare_data(df_ratiostopwords):
#     X = df_ratiostopwords.drop(['headline', 'clickbait', 'sensation_score',
#                    'sensation_reason', 'emotion', 'arousal', 'arousal_score',
#                    'arousal_reason', 'sensation'], axis=1)
#     y = df_ratiostopwords['sensation']
# 
#     # split dataset
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# 
#     numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
#     categorical_features = X.select_dtypes(include=['object', 'category']).columns
# 
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', StandardScaler(), numeric_features),
#             ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
#         ])
# 
#     pipeline = Pipeline(steps=[
#         ('preprocessor', preprocessor)
#     ])
# 
#     X_train_processed = pipeline.fit_transform(X_train)
# 
#     smote = SMOTE(random_state=42)
#     X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
# 
#     X_val_processed = pipeline.transform(X_val)
# 
#     return X_train_resampled, X_val_processed, y_train_resampled, y_val, pipeline

# prepare data
X_train, X_val, y_train, y_val, preprocessor = prepare_data(df_ratiostopwords)

# create XGBoost classifier
xgb_classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

param_dist_xgb = {
    'max_depth': [3, 6, 9],
    'min_child_weight': [1, 3],
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1]
}

# create K-Fold cross-validation object
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

random_search_xg = RandomizedSearchCV(
    estimator=xgb_classifier,
    param_distributions=param_dist_xgb,
    n_iter=100,
    cv=kfold,
    n_jobs=-1,
    verbose=2,
    scoring='f1',
    random_state=42
)

# fit on preprocessed training data
random_search_xg.fit(X_train, y_train)

print("Best parameters:", random_search_xg.best_params_)
print("Best cross-validation score:", random_search_xg.best_score_)

# use best parameters to create new model
best_params = random_search_xg.best_params_
best_model = xgb.XGBClassifier(
    **best_params,
    use_label_encoder=False,
    eval_metric='logloss',
    early_stopping_rounds=50  # Add early stopping here
)

# train the model with early stopping
best_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# use best model for prediction on validation set
y_val_pred = best_model.predict(X_val)

# print classification report
print("\nValidation Set Classification Report:")
print(classification_report(y_val, y_val_pred))

"""### ROC AUC"""

# ROC AUC Curve
fpr, tpr, _ = roc_curve(y_val, best_model.predict_proba(X_val)[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_val, best_model.predict_proba(X_val)[:, 1])
pr_auc = auc(recall, precision)

plt.figure(figsize=(10, 8))
plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

"""## Informality (Flesch-Kincaid Readability)"""



# dataframe of Flesch-Kincaid Readability
df_fkreadability = df_original.copy()

# function to Flesch-Kincaid Readability
def calculate_fk_readability(headline):
    return textstat.flesch_kincaid_grade(headline)

df_fkreadability['fk_readability'] = df_fkreadability['headline'].apply(calculate_fk_readability)
print(df_fkreadability[['headline', 'fk_readability']].head())

# kernel density estimation plot
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df_fkreadability, x='fk_readability', hue='sensation', shade=True)
plt.title('Distribution of Flesch-Kincaid Readability Scores')
plt.show()

# descriptive statistics
desc_stats = df_fkreadability.groupby('sensation')['fk_readability'].describe()
print("Descriptive Statistics:")
print(desc_stats)

# perform t-test
sensation_scores = df_fkreadability[df_fkreadability['sensation'] == 1]['fk_readability']
non_sensation_scores = df_fkreadability[df_fkreadability['sensation'] == 0]['fk_readability']
t_stat, p_value = stats.ttest_ind(sensation_scores, non_sensation_scores)

print(f"\nt-test results for Flesch-Kincaid Readability Scores:")
print(f"t-statistic: {t_stat}")
print(f"p-value: {p_value}")

# save .csv in utf-8
output_path = f'{base_dir}/df_fkreadability.csv'
df_fkreadability.to_csv(output_path, index=False, encoding='utf-8')

"""### XGBoost"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# base_dir = '/content/drive/My Drive/seperate_0731'
# 
# def prepare_data(df_fkreadability):
#     X = df_ratiostopwords.drop(['headline', 'clickbait', 'sensation_score',
#                    'sensation_reason', 'emotion', 'arousal', 'arousal_score',
#                    'arousal_reason', 'sensation'], axis=1)
#     y = df_fkreadability['sensation']
# 
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# 
#     numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
#     categorical_features = X.select_dtypes(include=['object', 'category']).columns
# 
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', StandardScaler(), numeric_features),
#             ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
#         ])
# 
#     pipeline = Pipeline(steps=[
#         ('preprocessor', preprocessor)
#     ])
# 
#     X_train_processed = pipeline.fit_transform(X_train)
# 
#     smote = SMOTE(random_state=42)
#     X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
# 
#     X_val_processed = pipeline.transform(X_val)
# 
#     return X_train_resampled, X_val_processed, y_train_resampled, y_val, pipeline

X_train, X_val, y_train, y_val, preprocessor = prepare_data(df_fkreadability)

# create XGBoost classifier
xgb_classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

param_dist_xgb = {
    'max_depth': [3, 6, 9],
    'min_child_weight': [1, 3],
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1]
}

# create K-Fold cross-validation object
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

random_search_xg = RandomizedSearchCV(
    estimator=xgb_classifier,
    param_distributions=param_dist_xgb,
    n_iter=100,
    cv=kfold,
    n_jobs=-1,
    verbose=2,
    scoring='f1',
    random_state=42
)

# fit on preprocessed training data
random_search_xg.fit(X_train, y_train)

print("Best parameters:", random_search_xg.best_params_)
print("Best cross-validation score:", random_search_xg.best_score_)

# use best parameters to create new model
best_params = random_search_xg.best_params_
best_model = xgb.XGBClassifier(
    **best_params,
    use_label_encoder=False,
    eval_metric='logloss',
    early_stopping_rounds=50  # Add early stopping here
)

# train the model with early stopping
best_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# use best model for prediction on validation set
y_val_pred = best_model.predict(X_val)

# print classification report
print("\nValidation Set Classification Report:")
print(classification_report(y_val, y_val_pred))

"""### ROC AUC"""

# ROC AUC Curve
fpr, tpr, _ = roc_curve(y_val, best_model.predict_proba(X_val)[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_val, best_model.predict_proba(X_val)[:, 1])
pr_auc = auc(recall, precision)

plt.figure(figsize=(10, 8))
plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

"""## Sentence Subjectivity and Objectivity Evaluation"""

# VADER
sid = SentimentIntensityAnalyzer()

# function to define objectively and objectively
def get_subjectivity_objectivity(text):
    sentiment_scores = sid.polarity_scores(text)
    subjectivity = sentiment_scores['pos'] + sentiment_scores['neg']
    objectivity = sentiment_scores['neu']
    return subjectivity, objectivity

# calculate score
df_subobjectivity = df_original.copy()
df_subobjectivity['subjectivity'], df_subobjectivity['objectivity'] = zip(*df_fkreadability['headline'].apply(get_subjectivity_objectivity))

print(df_subobjectivity[['headline', 'subjectivity', 'objectivity']])

# kernel density estimation plot
plt.figure(figsize=(12, 5))

# subjectivity kernel density estimation plot
plt.subplot(1, 2, 1)
sns.kdeplot(data=df_subobjectivity, x='subjectivity', hue='sensation', shade=True)
plt.title('Distribution of Subjectivity Scores')
plt.xlabel('Subjectivity Score')
plt.ylabel('Density')

# objectivity kernel density estimation plot
plt.subplot(1, 2, 2)
sns.kdeplot(data=df_subobjectivity, x='objectivity', hue='sensation', shade=True)
plt.title('Distribution of Objectivity Scores')
plt.xlabel('Objectivity Score')
plt.ylabel('Density')

plt.tight_layout()
plt.show()

# descriptive statistics
for metric in ['subjectivity', 'objectivity']:
    print(f"\nDescriptive Statistics for {metric.capitalize()}:")
    desc_stats = df_subobjectivity.groupby('sensation')[metric].describe()
    print(desc_stats)

# t-test function
def perform_t_test(data, column):
    sensation = data[data['sensation'] == 1][column]
    non_sensation = data[data['sensation'] == 0][column]
    t_stat, p_value = stats.ttest_ind(sensation, non_sensation)

    print(f"\nt-test results for {column}:")
    print(f"t-statistic: {t_stat}")
    print(f"p-value: {p_value}")

# perform t-test
perform_t_test(df_subobjectivity, 'subjectivity')
perform_t_test(df_subobjectivity, 'objectivity')

# save .csv in utf-8
output_path = f'{base_dir}/df_subobjectivity.csv'
df_subobjectivity.to_csv(output_path, index=False, encoding='utf-8')

"""### XGBoost"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# base_dir = '/content/drive/My Drive/seperate_0731'
# 
# def prepare_data(df_subobjectivity):
#     X = df_subobjectivity.drop(['headline', 'clickbait', 'sensation_score',
#                    'sensation_reason', 'emotion', 'arousal', 'arousal_score',
#                    'arousal_reason', 'sensation'], axis=1)
#     y = df_subobjectivity['sensation']
# 
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# 
#     numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
#     categorical_features = X.select_dtypes(include=['object', 'category']).columns
# 
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', StandardScaler(), numeric_features),
#             ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
#         ])
# 
#     pipeline = Pipeline(steps=[
#         ('preprocessor', preprocessor)
#     ])
# 
#     X_train_processed = pipeline.fit_transform(X_train)
# 
#     smote = SMOTE(random_state=42)
#     X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
# 
#     X_val_processed = pipeline.transform(X_val)
# 
#     return X_train_resampled, X_val_processed, y_train_resampled, y_val, pipeline

X_train, X_val, y_train, y_val, preprocessor = prepare_data(df_subobjectivity)

# create XGBoost classifier
xgb_classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

param_dist_xgb = {
    'max_depth': [3, 6, 9],
    'min_child_weight': [1, 3],
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1]
}

# create K-Fold cross-validation object
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

random_search_xg = RandomizedSearchCV(
    estimator=xgb_classifier,
    param_distributions=param_dist_xgb,
    n_iter=100,
    cv=kfold,
    n_jobs=-1,
    verbose=2,
    scoring='f1',
    random_state=42
)

# fit on preprocessed training data
random_search_xg.fit(X_train, y_train)

print("Best parameters:", random_search_xg.best_params_)
print("Best cross-validation score:", random_search_xg.best_score_)

# use best parameters to create new model
best_params = random_search_xg.best_params_
best_model = xgb.XGBClassifier(
    **best_params,
    use_label_encoder=False,
    eval_metric='logloss',
    early_stopping_rounds=50
)

# train the model with early stopping
best_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# use best model for prediction on validation set
y_val_pred = best_model.predict(X_val)

# print classification report
print("\nValidation Set Classification Report:")
print(classification_report(y_val, y_val_pred))

"""### ROC AUC"""

# ROC AUC Curve
fpr, tpr, _ = roc_curve(y_val, best_model.predict_proba(X_val)[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_val, best_model.predict_proba(X_val)[:, 1])
pr_auc = auc(recall, precision)

plt.figure(figsize=(10, 8))
plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

"""## Sentiment Analysis"""

# Vader
sid = SentimentIntensityAnalyzer()

# count sentiment scores
def get_sentiment(text):
    sentiment_scores = sid.polarity_scores(text)
    return sentiment_scores['neg'], sentiment_scores['neu'], sentiment_scores['pos'], sentiment_scores['compound']

# calculate sentiment score
df_sentiment = df_original.copy()
df_sentiment[['neg', 'neu', 'pos', 'compound']] = df_sentiment['headline'].apply(lambda x: pd.Series(get_sentiment(x)))

print(df_sentiment[['headline', 'neg', 'neu', 'pos', 'compound']])

# kernel density estimation plot
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Distribution of Sentiment Scores', fontsize=16)

sentiment_types = ['neg', 'neu', 'pos', 'compound']

for i, sentiment in enumerate(sentiment_types):
    ax = axes[i//2, i%2]
    sns.kdeplot(data=df_sentiment, x=sentiment, hue='sensation', shade=True, ax=ax)
    ax.set_title(f'{sentiment.capitalize()} Sentiment Score')
    ax.set_xlabel('Score')
    ax.set_ylabel('Density')

plt.tight_layout()
plt.show()

# descriptive statistics
for sentiment in sentiment_types:
    print(f"\nDescriptive Statistics for {sentiment.capitalize()} Sentiment:")
    desc_stats = df_sentiment.groupby('sensation')[sentiment].describe()
    print(desc_stats)

# t-test function
def perform_t_test(data, column):
    sensation = data[data['sensation'] == 1][column]
    non_sensation = data[data['sensation'] == 0][column]
    t_stat, p_value = stats.ttest_ind(sensation, non_sensation)

    print(f"\nt-test results for {column} Sentiment:")
    print(f"t-statistic: {t_stat}")
    print(f"p-value: {p_value}")

# perform t-test
for sentiment in sentiment_types:
    perform_t_test(df_sentiment, sentiment)

# save .csv in utf-8
output_path = f'{base_dir}/df_sentiment.csv'
df_sentiment.to_csv(output_path, index=False, encoding='utf-8')

"""### XGBoost"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# base_dir = '/content/drive/My Drive/seperate_0731'
# 
# def prepare_data(df_sentiment):
#     X = df_sentiment.drop(['headline', 'clickbait', 'sensation_score',
#                    'sensation_reason', 'emotion', 'arousal', 'arousal_score',
#                    'arousal_reason', 'sensation'], axis=1)
#     y = df_sentiment['sensation']
# 
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# 
#     numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
#     categorical_features = X.select_dtypes(include=['object', 'category']).columns
# 
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', StandardScaler(), numeric_features),
#             ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
#         ])
# 
#     pipeline = Pipeline(steps=[
#         ('preprocessor', preprocessor)
#     ])
# 
#     X_train_processed = pipeline.fit_transform(X_train)
# 
#     smote = SMOTE(random_state=42)
#     X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
# 
#     X_val_processed = pipeline.transform(X_val)
# 
#     return X_train_resampled, X_val_processed, y_train_resampled, y_val, pipeline

X_train, X_val, y_train, y_val, preprocessor = prepare_data(df_sentiment)

xgb_classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

param_dist_xgb = {
    'max_depth': [3, 6, 9],
    'min_child_weight': [1, 3],
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1]
}

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

random_search_xg = RandomizedSearchCV(
    estimator=xgb_classifier,
    param_distributions=param_dist_xgb,
    n_iter=100,
    cv=kfold,
    n_jobs=-1,
    verbose=2,
    scoring='f1',
    random_state=42
)

# fit on preprocessed training data
random_search_xg.fit(X_train, y_train)

print("Best parameters:", random_search_xg.best_params_)
print("Best cross-validation score:", random_search_xg.best_score_)

# use best parameters to create new model
best_params = random_search_xg.best_params_
best_model = xgb.XGBClassifier(
    **best_params,
    use_label_encoder=False,
    eval_metric='logloss',
    early_stopping_rounds=50
)

# train the model with early stopping
best_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# use best model for prediction on validation set
y_val_pred = best_model.predict(X_val)

print("\nValidation Set Classification Report:")
print(classification_report(y_val, y_val_pred))

"""### ROC AUC"""

# ROC AUC Curve
fpr, tpr, _ = roc_curve(y_val, best_model.predict_proba(X_val)[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_val, best_model.predict_proba(X_val)[:, 1])
pr_auc = auc(recall, precision)

plt.figure(figsize=(10, 8))
plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

"""## Elongated Words"""

# Elongated words dataframe
df_elongated = df_original.copy()

def count_elongated_words(text):
    elongated_pattern = re.compile(r'\b\w*(\w)\1{3,}\w*\b')
    elongated_words = elongated_pattern.findall(text)
    return len(elongated_words)

df_elongated['elongated_word_count'] = df_elongated['headline'].apply(count_elongated_words)

print(df_elongated.head())

# kernel density estimation plot
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df_elongated, x='elongated_word_count', hue='sensation', shade=True)
plt.title('Distribution of Elongated Word Count', fontsize=16)
plt.xlabel('Elongated Word Count', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(title='Sensation', labels=['Non-Sensation', 'Sensation'])
plt.tight_layout()
plt.show()

# descriptive statistics
desc_stats = df_elongated.groupby('sensation')['elongated_word_count'].describe()
print("Descriptive Statistics for Elongated Word Count:")
print(desc_stats)

# # perform t-test
def perform_t_test(data, column):
    sensation = data[data['sensation'] == 1][column]
    non_sensation = data[data['sensation'] == 0][column]
    t_stat, p_value = stats.ttest_ind(sensation, non_sensation)

    print(f"\nt-test results for {column}:")
    print(f"t-statistic: {t_stat}")
    print(f"p-value: {p_value}")

# perform t-test
perform_t_test(df_elongated, 'elongated_word_count')

# calculate the proportion of titles with elongated words in each category
def calculate_proportion(data, column):
    total = len(data)
    count_with_elongated = len(data[data[column] > 0])
    return count_with_elongated / total

sensation_prop = calculate_proportion(df_elongated[df_elongated['sensation'] == 1], 'elongated_word_count')
non_sensation_prop = calculate_proportion(df_elongated[df_elongated['sensation'] == 0], 'elongated_word_count')

print(f"\nProportion of headlines with elongated words:")
print(f"Sensation: {sensation_prop:.2%}")
print(f"Non-Sensation: {non_sensation_prop:.2%}")

# save .csv in utf-8
output_path = f'{base_dir}/df_elongated.csv'
df_elongated.to_csv(output_path, index=False, encoding='utf-8')

"""### XGBoost"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# 
# base_dir = '/content/drive/My Drive/seperate_0731'
# 
# def prepare_data(df_elongated):
#     X = df_elongated.drop(['headline', 'clickbait', 'sensation_score',
#                    'sensation_reason', 'emotion', 'arousal', 'arousal_score',
#                    'arousal_reason', 'sensation'], axis=1)
#     y = df_elongated['sensation']
# 
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# 
#     numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
#     categorical_features = X.select_dtypes(include=['object', 'category']).columns
# 
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', StandardScaler(), numeric_features),
#             ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
#         ])
# 
#     pipeline = Pipeline(steps=[
#         ('preprocessor', preprocessor)
#     ])
# 
#     # fit and transform on train set
#     X_train_processed = pipeline.fit_transform(X_train)
# 
#     # SMOTE
#     smote = SMOTE(random_state=42)
#     X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
# 
#     # transform on validation set
#     X_val_processed = pipeline.transform(X_val)
# 
#     return X_train_resampled, X_val_processed, y_train_resampled, y_val, pipeline

X_train, X_val, y_train, y_val, preprocessor = prepare_data(df_elongated)

xgb_classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

param_dist_xgb = {
    'max_depth': [3, 6, 9],
    'min_child_weight': [1, 3],
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1]
}

# K-Fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

random_search_xg = RandomizedSearchCV(
    estimator=xgb_classifier,
    param_distributions=param_dist_xgb,
    n_iter=100,
    cv=kfold,
    n_jobs=-1,
    verbose=2,
    scoring='f1',
    random_state=42
)

# fit on preprocessed training data
random_search_xg.fit(X_train, y_train)

print("Best parameters:", random_search_xg.best_params_)
print("Best cross-validation score:", random_search_xg.best_score_)

# use best parameters to create new model
best_params = random_search_xg.best_params_
best_model = xgb.XGBClassifier(
    **best_params,
    use_label_encoder=False,
    eval_metric='logloss',
    early_stopping_rounds=50  # Add early stopping here
)

# train the model with early stopping
best_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# use best model for prediction on validation set
y_val_pred = best_model.predict(X_val)

print("\nValidation Set Classification Report:")
print(classification_report(y_val, y_val_pred))

"""### ROC AUC"""

# ROC AUC Curve
fpr, tpr, _ = roc_curve(y_val, best_model.predict_proba(X_val)[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_val, best_model.predict_proba(X_val)[:, 1])
pr_auc = auc(recall, precision)

plt.figure(figsize=(10, 8))
plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

"""## Punctuation"""

df_punctuation = df_original.copy()

def count_punctuation(text):
    contracted_word_forms_count = len(re.findall(r"\b\w+['']\w+\b", text))
    single_quotes_count = text.count("'") - contracted_word_forms_count  # remove single quotes from abbreviations
    punctuation_counts = {
        'currency_count': len(re.findall(r'[$€£]', text)),
        'exclamation_count': text.count('!'),
        'question_mark_count': text.count('?'),
        'ellipsis_count': text.count('…') + text.count('...'),
        'emphasis_count': len(re.findall(r':\*\*\*', text)),
        'multiple_exclamation_count': len(re.findall(r'!!!', text)),
        'single_quotes_count': single_quotes_count,
        'double_quotes_count': text.count('"'),
        'contracted_word_forms_count': contracted_word_forms_count
    }
    return pd.Series(punctuation_counts)

df_punctuation[['currency_count', 'exclamation_count', 'question_mark_count',
                'ellipsis_count', 'emphasis_count', 'multiple_exclamation_count',
                'single_quotes_count', 'double_quotes_count',
                'contracted_word_forms_count']] = df_punctuation['headline'].apply(count_punctuation)

print(df_punctuation.head())

total_currency = df_punctuation['currency_count'].sum()
total_exclamation = df_punctuation['exclamation_count'].sum()
total_question_mark = df_punctuation['question_mark_count'].sum()
total_ellipsis = df_punctuation['ellipsis_count'].sum()
total_emphasis = df_punctuation['emphasis_count'].sum()
total_multiple_exclamation = df_punctuation['multiple_exclamation_count'].sum()
total_single_quotes = df_punctuation['single_quotes_count'].sum()
total_double_quotes = df_punctuation['double_quotes_count'].sum()
total_quotes = total_single_quotes + total_double_quotes
total_contracted_word_forms = df_punctuation['contracted_word_forms_count'].sum()

print(f"Total number of currency signs: {total_currency}")
print(f"Total number of exclamation marks: {total_exclamation}")
print(f"Total number of question marks: {total_question_mark}")
print(f"Total number of ellipses: {total_ellipsis}")
print(f"Total number of emphasis marks: {total_emphasis}")
print(f"Total number of multiple exclamation marks: {total_multiple_exclamation}")
print(f"Total number of single quotes: {total_single_quotes}")
print(f"Total number of double quotes: {total_double_quotes}")
print(f"Total number of quotes: {total_quotes}")
print(f"Total number of contracted word forms: {total_contracted_word_forms}")

def analyze_punctuation(df, column):
    #  kernel density estimation plot
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x=column, hue='sensation', shade=True)
    plt.title(f'Distribution of {column}', fontsize=16)
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(title='Sensation', labels=['Non-Sensation', 'Sensation'])
    plt.tight_layout()
    plt.show()

    # descriptive statistics
    desc_stats = df.groupby('sensation')[column].describe()
    print(f"\nDescriptive Statistics for {column}:")
    print(desc_stats)

    # t-test
    sensation = df[df['sensation'] == 1][column]
    non_sensation = df[df['sensation'] == 0][column]
    t_stat, p_value = stats.ttest_ind(sensation, non_sensation)

    print(f"\nt-test results for {column}:")
    print(f"t-statistic: {t_stat}")
    print(f"p-value: {p_value}")

    # calculate the proportion of titles in each category that have this punctuation mark
    sensation_prop = (sensation > 0).mean()
    non_sensation_prop = (non_sensation > 0).mean()

    print(f"\nProportion of headlines with {column}:")
    print(f"Sensation: {sensation_prop:.2%}")
    print(f"Non-Sensation: {non_sensation_prop:.2%}")

# analyze each punctuation variable
punctuation_columns = ['currency_count', 'exclamation_count', 'question_mark_count',
                       'ellipsis_count', 'emphasis_count', 'multiple_exclamation_count',
                       'single_quotes_count', 'double_quotes_count',
                       'contracted_word_forms_count']

for column in punctuation_columns:
    analyze_punctuation(df_punctuation, column)

# save .csv in utf-8
output_path = f'{base_dir}/df_punctuation.csv'
df_punctuation.to_csv(output_path, index=False, encoding='utf-8')

"""## Variance Threshold"""

# select the columns to filter
punctuation_columns = ['currency_count', 'exclamation_count', 'question_mark_count',
                       'ellipsis_count', 'emphasis_count', 'multiple_exclamation_count',
                       'single_quotes_count', 'double_quotes_count',
                       'contracted_word_forms_count']

# create the feature matrix
X = df_punctuation[punctuation_columns]

# initialize VarianceThreshold
selector = VarianceThreshold()

# fit the data and transform it
X_selected = selector.fit_transform(X)

# get the boolean mask of selected features
support = selector.get_support()

# get the names of selected features
selected_features = X.columns[support]

print("Selected features:")
print(selected_features)

# print the variance of each feature
feature_variances = selector.variances_
for feature, variance in zip(X.columns, feature_variances):
    print(f"{feature}: {variance}")

# create a new DataFrame with only the selected features
df_selected_punctuation = df_punctuation[list(selected_features) + ['sensation', 'headline']]

print("\nSelected features DataFrame:")
print(df_selected_punctuation.head())

# save .csv in utf-8
output_path = f'{base_dir}/df_selected_punctuation.csv'
df_punctuation.to_csv(output_path, index=False, encoding='utf-8')

"""### XGBoost"""

# modify the prepare_data function for df_selected_punctuation
def prepare_data_punctuation(df_now):
    X = df_now.drop(['sensation', 'headline'], axis=1, errors='ignore')
    y = df_now['sensation']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    preprocessor = StandardScaler()

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])

    # fit and transform the training set
    X_train_processed = pipeline.fit_transform(X_train)

    # transform the validation set
    X_val_processed = pipeline.transform(X_val)

    return X_train_processed, X_val_processed, y_train, y_val, pipeline

X_train, X_val, y_train, y_val, preprocessor = prepare_data_punctuation(df_selected_punctuation)

xgb_classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

param_dist_xgb = {
    'max_depth': [3, 6, 9],
    'min_child_weight': [1, 3],
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1]
}

# k-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

random_search_xg = RandomizedSearchCV(
    estimator=xgb_classifier,
    param_distributions=param_dist_xgb,
    n_iter=100,
    cv=kfold,
    n_jobs=-1,
    verbose=2,
    scoring='f1',
    random_state=42
)

# fit on the preprocessed training data
random_search_xg.fit(X_train, y_train)

print("Best parameters:", random_search_xg.best_params_)
print("Best cross-validation score:", random_search_xg.best_score_)

# use best parameters to create new model
best_params = random_search_xg.best_params_
best_model = xgb.XGBClassifier(
    **best_params,
    use_label_encoder=False,
    eval_metric='logloss',
    early_stopping_rounds=50
)

# train the model with early stopping
best_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# use best model for prediction on validation set
y_val_pred = best_model.predict(X_val)

print("\nValidation Set Classification Report:")
print(classification_report(y_val, y_val_pred))

"""### ROC AUC"""

# ROC AUC Curve
fpr, tpr, _ = roc_curve(y_val, best_model.predict_proba(X_val)[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_val, best_model.predict_proba(X_val)[:, 1])
pr_auc = auc(recall, precision)

plt.figure(figsize=(10, 8))
plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

"""## TF-IDF

### with Stopwords
"""

# dataframe TF-IDF with stopwords
df_tfidf = df_original.copy()

# calculate TF-IDF features
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df_tfidf['headline'])
tfidf_feature_names = vectorizer.get_feature_names_out()
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f'tfidf_{feature}' for feature in tfidf_feature_names])

# Add TF-IDF features back to df_tfidf
df_tfidf = pd.concat([df_tfidf, tfidf_df], axis=1)

print(df_tfidf.head())

# save .csv in utf-8
output_path = f'{base_dir}/df_tfidf.csv'
df_tfidf.to_csv(output_path, index=False, encoding='utf-8')

# load the dataframe
input_path = f'{base_dir}/df_tfidf.csv'
df_tfidf = pd.read_csv(input_path, encoding='utf-8')

# initialize variancethreshold with a lower threshold
variance_threshold = VarianceThreshold(threshold=0.001)

# select tf-idf feature columns
tfidf_columns = [col for col in df_tfidf.columns if col.startswith('tfidf_')]

# apply variance threshold for feature selection
tfidf_features = df_tfidf[tfidf_columns]
selected_features = variance_threshold.fit_transform(tfidf_features)
selected_feature_names = [tfidf_columns[i] for i in range(len(tfidf_columns)) if variance_threshold.variances_[i] >= 0.001]

# add selected features back to the original dataframe
df_selected_tfidf = pd.DataFrame(selected_features, columns=selected_feature_names)
df_tfidf_selected = pd.concat([df_tfidf.drop(tfidf_columns, axis=1), df_selected_tfidf], axis=1)

print(df_tfidf_selected.head())

# save the selected dataframe as a csv file
output_selected_path = f'{base_dir}/df_tfidf_selected.csv'
df_tfidf_selected.to_csv(output_selected_path, index=False, encoding='utf-8')

"""### XGBoost"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# base_dir = '/content/drive/My Drive/seperate_0731'
# 
# def prepare_data(df_sentiment):
#     X = df_sentiment.drop(['headline', 'clickbait', 'sensation_score',
#                    'sensation_reason', 'emotion', 'arousal', 'arousal_score',
#                    'arousal_reason', 'sensation'], axis=1)
#     y = df_sentiment['sensation']
# 
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# 
#     numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
#     categorical_features = X.select_dtypes(include=['object', 'category']).columns
# 
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', StandardScaler(), numeric_features),
#             ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
#         ])
# 
#     pipeline = Pipeline(steps=[
#         ('preprocessor', preprocessor)
#     ])
# 
#     X_train_processed = pipeline.fit_transform(X_train)
# 
#     smote = SMOTE(random_state=42)
#     X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
# 
#     X_val_processed = pipeline.transform(X_val)
# 
#     return X_train_resampled, X_val_processed, y_train_resampled, y_val, pipeline

X_train, X_val, y_train, y_val, preprocessor = prepare_data(df_tfidf_selected)

xgb_classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

param_dist_xgb = {
    'max_depth': [3, 6, 9],
    'min_child_weight': [1, 3],
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1]
}

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

random_search_xg = RandomizedSearchCV(
    estimator=xgb_classifier,
    param_distributions=param_dist_xgb,
    n_iter=100,
    cv=kfold,
    n_jobs=-1,
    verbose=2,
    scoring='f1',
    random_state=42
)

# fit on preprocessed training data
random_search_xg.fit(X_train, y_train)

print("Best parameters:", random_search_xg.best_params_)
print("Best cross-validation score:", random_search_xg.best_score_)

# use best parameters to create new model
best_params = random_search_xg.best_params_
best_model = xgb.XGBClassifier(
    **best_params,
    use_label_encoder=False,
    eval_metric='logloss',
    early_stopping_rounds=50  # Add early stopping here
)

# train the model with early stopping
best_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# use best model for prediction on validation set
y_val_pred = best_model.predict(X_val)

print("\nValidation Set Classification Report:")
print(classification_report(y_val, y_val_pred))

"""### ROC AUC"""

# ROC AUC Curve
fpr, tpr, _ = roc_curve(y_val, best_model.predict_proba(X_val)[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_val, best_model.predict_proba(X_val)[:, 1])
pr_auc = auc(recall, precision)

plt.figure(figsize=(10, 8))
plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

"""### w/o stopwords"""

# load the spacy model for English language
nlp = spacy.load('en_core_web_sm')

# function to remove stopwords using spacy
def remove_stopwords(text):
    doc = nlp(text)
    return " ".join([token.text for token in doc if not token.is_stop])

# dataframe TF-IDF with stopwords removed
df_tfidf_wostop = df_original.copy()

# apply the function to remove stopwords
df_tfidf_wostop['cleaned_headline'] = df_tfidf_wostop['headline'].apply(remove_stopwords)

# calculate TF-IDF features, now without stopwords
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df_tfidf_wostop['cleaned_headline'])
tfidf_feature_names = vectorizer.get_feature_names_out()
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f'tfidf_{feature}' for feature in tfidf_feature_names])

# add TF-IDF features back to df_tfidf_wostop
df_tfidf_wostop = pd.concat([df_tfidf_wostop, tfidf_df], axis=1)

print(df_tfidf_wostop.head())

# save .csv in utf-8
output_path = f'{base_dir}/df_tfidf_wostop.csv'
df_tfidf_wostop.to_csv(output_path, index=False, encoding='utf-8')

import pandas as pd
from sklearn.feature_selection import VarianceThreshold

# load the dataframe
input_path = f'{base_dir}/df_tfidf_wostop.csv'
df_tfidf_wostop = pd.read_csv(input_path, encoding='utf-8')

# initialize VarianceThreshold with a lower threshold
variance_threshold = VarianceThreshold(threshold=0.001)

# select TF-IDF feature columns
tfidf_columns = [col for col in df_tfidf_wostop.columns if col.startswith('tfidf_')]

# apply Variance Threshold for feature selection
tfidf_features = df_tfidf_wostop[tfidf_columns]
selected_features = variance_threshold.fit_transform(tfidf_features)
selected_feature_names = [tfidf_columns[i] for i in range(len(tfidf_columns)) if variance_threshold.variances_[i] >= 0.001]

# create a DataFrame for the selected features
df_selected_tfidf = pd.DataFrame(selected_features, columns=selected_feature_names)

# add selected features back to the original dataframe
df_tfidf_wostop_selected = pd.concat([df_tfidf_wostop.drop(tfidf_columns, axis=1), df_selected_tfidf], axis=1)

print(df_tfidf_wostop_selected.head())

# save the selected dataframe as a CSV file
output_selected_path = f'{base_dir}/df_tfidf_wostop_selected.csv'
df_tfidf_wostop_selected.to_csv(output_selected_path, index=False, encoding='utf-8')

"""### XGBoost"""

base_dir = '/content/drive/My Drive/seperate_0731'

def prepare_data(df_sentiment):
    X = df_sentiment.drop(['headline', 'clickbait', 'sensation_score',
                   'sensation_reason', 'emotion', 'arousal', 'arousal_score',
                   'arousal_reason', 'sensation'], axis=1)
    y = df_sentiment['sensation']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])

    X_train_processed = pipeline.fit_transform(X_train)

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)

    X_val_processed = pipeline.transform(X_val)

    return X_train_resampled, X_val_processed, y_train_resampled, y_val, pipeline

X_train, X_val, y_train, y_val, preprocessor = prepare_data(df_tfidf_wostop_selected)

xgb_classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

param_dist_xgb = {
    'max_depth': [3, 6, 9],
    'min_child_weight': [1, 3],
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0, 0.1]
}

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

random_search_xg = RandomizedSearchCV(
    estimator=xgb_classifier,
    param_distributions=param_dist_xgb,
    n_iter=100,
    cv=kfold,
    n_jobs=-1,
    verbose=2,
    scoring='f1',
    random_state=42
)

# fit on preprocessed training data
random_search_xg.fit(X_train, y_train)

print("Best parameters:", random_search_xg.best_params_)
print("Best cross-validation score:", random_search_xg.best_score_)

# use best parameters to create new model
best_params = random_search_xg.best_params_
best_model = xgb.XGBClassifier(
    **best_params,
    use_label_encoder=False,
    eval_metric='logloss',
    early_stopping_rounds=50  # Add early stopping here
)

# train the model with early stopping
best_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# use best model for prediction on validation set
y_val_pred = best_model.predict(X_val)

print("\nValidation Set Classification Report:")
print(classification_report(y_val, y_val_pred))

"""### ROC AUC"""

# ROC AUC Curve
fpr, tpr, _ = roc_curve(y_val, best_model.predict_proba(X_val)[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_val, best_model.predict_proba(X_val)[:, 1])
pr_auc = auc(recall, precision)

plt.figure(figsize=(10, 8))
plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

"""## Syntactic Ngrams"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# 
# nlp = spacy.load("en_core_web_sm")
# 
# df_4grams_syntactic = df_original.copy()
# 
# # extract dependency tags
# def extract_dep_tags(doc):
#     return [token.dep_ for token in doc]
# 
# # apply function and calculate stats
# df_4grams_syntactic['dep_tags'] = df_4grams_syntactic['headline'].apply(lambda x: extract_dep_tags(nlp(x)))
# 
# # calculate dependency stats
# dep_counts = df_4grams_syntactic['dep_tags'].apply(len)
# 
# min_deps = dep_counts.min()
# max_deps = dep_counts.max()
# avg_deps = dep_counts.mean()
# 
# print("\ndependency stats:")
# print(f"min dependencies: {min_deps}")
# print(f"max dependencies: {max_deps}")
# print(f"average dependencies: {avg_deps:.2f}")
# 
# # calculate syntactic tree depth
# def tree_depth(doc):
#     def token_depth(token):
#         depth = 0
#         while token.head != token:
#             token = token.head
#             depth += 1
#         return depth
# 
#     return max(token_depth(token) for token in doc)
# 
# # apply function to dataframe
# df_4grams_syntactic['tree_depth'] = df_4grams_syntactic['headline'].apply(lambda x: tree_depth(nlp(x)))
# 
# min_depth = df_4grams_syntactic['tree_depth'].min()
# max_depth = df_4grams_syntactic['tree_depth'].max()
# avg_depth = df_4grams_syntactic['tree_depth'].mean()
# 
# print("\ntree depth stats:")
# print(f"min tree depth: {min_depth}")
# print(f"max tree depth: {max_depth}")
# print(f"average tree depth: {avg_depth:.2f}")

output_path = f'{base_dir}/df_depth_syntactic.csv'
df_4grams_syntactic.to_csv(output_path, index=False, encoding='utf-8')

print(f"\nsave to: {output_path}")

# del dep_counts, min_deps, max_deps, avg_deps
# gc.collect()

# Commented out IPython magic to ensure Python compatibility.
# %%time
# 
# n = int(4)
# 
# # generate dependency n-grams and count them
# def generate_syntactic_ngrams(dep_tags, n):
#     dep_ngrams = list(ngrams(dep_tags, n))
#     return Counter(dep_ngrams)
# 
# # generate n-gram features for each headline
# def add_syntactic_ngram_features(df, n):
#     df[f'syntactic_{n}grams'] = df['dep_tags'].apply(lambda tags: generate_syntactic_ngrams(tags, n))
#     return df
# 
# # expand n-gram features into separate columns
# def expand_ngrams(df, n, prefix='syntactic'):
#     ngram_col = f'{prefix}_{n}grams'
#     all_ngrams = set(ngram for ngram_counter in df[ngram_col] for ngram in ngram_counter)
#     new_columns = {f'{ngram_col}_{"_".join(ngram)}': df[ngram_col].apply(lambda x: x[ngram] if ngram in x else 0)
#                    for ngram in all_ngrams}
#     df = df.drop(columns=[ngram_col])
#     df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
#     return df
# 
# # generate and expand syntactic n-gram features
# df_4grams_syntactic = add_syntactic_ngram_features(df_4grams_syntactic, n)
# df_4grams_syntactic = expand_ngrams(df_4grams_syntactic, n)
# 
# # print the generated results
# print(df_4grams_syntactic.head())
#

# Commented out IPython magic to ensure Python compatibility.
# %%time
# 
# output_path = f'{base_dir}/df_4grams_syntactic.csv'
# df_4grams_syntactic.to_csv(output_path, index=False, encoding='utf-8')

# del generate_syntactic_ngrams, add_syntactic_ngram_features, expand_ngrams
# gc.collect()

variance_threshold = VarianceThreshold(threshold=0.001)

numeric_columns = df_4grams_syntactic.select_dtypes(include=['int64', 'float64']).columns

numeric_features = df_4grams_syntactic[numeric_columns]
selected_features = variance_threshold.fit_transform(numeric_features)

# get the mask of selected features
selected_mask = variance_threshold.get_support()

# use the mask to select feature names
selected_feature_names = numeric_columns[selected_mask]

df_selected_numeric_syntactic = pd.DataFrame(selected_features, columns=selected_feature_names)
df_selected_syntactic = pd.concat([df_4grams_syntactic.drop(numeric_columns, axis=1), df_selected_numeric_syntactic], axis=1)

print(f"Original features: {len(numeric_columns)}")
print(f"Selected features: {len(selected_feature_names)}")
print(f"Removed features: {len(numeric_columns) - len(selected_feature_names)}")

print(df_selected_syntactic.head())

output_selected_path = f'{base_dir}/df_selected_features_syntactic.csv'
df_selected_syntactic.to_csv(output_selected_path, index=False, encoding='utf-8')
print(f"\nsave to: {output_selected_path}")

# del numeric_features, selected_features
# gc.collect()

"""### XGBoost"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# 
# base_dir = '/content/drive/My Drive/seperate_0731'
# 
# 
# df_selected_features_syntactic = pd.read_csv(f'{base_dir}/df_selected_features_syntactic.csv')
# df_4grams_syntactic = pd.read_csv(f'{base_dir}/df_4grams_syntactic.csv')
# 
# def prepare_data(df_4grams_syntactic):
#     X = df_4grams_syntactic.drop(['headline', 'clickbait', 'sensation_score',
#                    'sensation_reason', 'emotion', 'arousal', 'arousal_score',
#                    'arousal_reason', 'sensation'], axis=1)
#     y = df_4grams_syntactic['sensation']
# 
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# 
#     numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
#     categorical_features = X.select_dtypes(include=['object', 'category']).columns
# 
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', StandardScaler(), numeric_features),
#             ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
#         ])
# 
#     pipeline = Pipeline(steps=[
#         ('preprocessor', preprocessor)
#     ])
# 
#     X_train_processed = pipeline.fit_transform(X_train)
# 
#     smote = SMOTE(random_state=42)
#     X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
# 
#     X_val_processed = pipeline.transform(X_val)
# 
#     return X_train_resampled, X_val_processed, y_train_resampled, y_val, pipeline

# Commented out IPython magic to ensure Python compatibility.
# %%time
# 
# X_train, X_val, y_train, y_val, preprocessor = prepare_data(df_selected_features_syntactic)
# 
# xgb_classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
# 
# param_dist_xgb = {
#     'max_depth': [3, 6, 9],
#     'min_child_weight': [1, 3],
#     'n_estimators': [100, 300, 500],
#     'learning_rate': [0.01, 0.1, 0.3],
#     'subsample': [0.8, 1.0],
#     'colsample_bytree': [0.8, 1.0],
#     'gamma': [0, 0.1]
# }
# 
# kfold = KFold(n_splits=5, shuffle=True, random_state=42)
# 
# random_search_xg = RandomizedSearchCV(
#     estimator=xgb_classifier,
#     param_distributions=param_dist_xgb,
#     n_iter=100,
#     cv=kfold,
#     n_jobs=-1,
#     verbose=2,
#     scoring='f1',
#     random_state=42
# )
# 
# random_search_xg.fit(X_train, y_train)
# 
# print("Best parameters:", random_search_xg.best_params_)
# print("Best cross-validation score:", random_search_xg.best_score_)
# 
# best_params = random_search_xg.best_params_
# best_model = xgb.XGBClassifier(
#     **best_params,
#     use_label_encoder=False,
#     eval_metric='logloss',
#     early_stopping_rounds=50)
# 
# best_model.fit(
#     X_train, y_train,
#     eval_set=[(X_val, y_val)],
#     verbose=False
# )
# 
# y_val_pred = best_model.predict(X_val)
# 
# print("\nValidation Set Classification Report:")
# print(classification_report(y_val, y_val_pred))

"""### ROC AUC"""

# ROC AUC Curve
fpr, tpr, _ = roc_curve(y_val, best_model.predict_proba(X_val)[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_val, best_model.predict_proba(X_val)[:, 1])
pr_auc = auc(recall, precision)

plt.figure(figsize=(10, 8))
plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# del df_selected_features_syntactic, df_4grams_syntactic, X_train, X_val, y_train, y_val, random_search_xg, xgb_classifier, best_model, y_val_pred, y_val_pred_proba, fpr, tpr, thresholds, cm
# gc.collect()

# print("done")

!pip install pipdeptree
!pipdeptree

import sys

for name, module in sys.modules.items():
    if hasattr(module, '__file__') and module.__file__:
        print(name)

!pip list

!pip freeze
!pip freeze > requirements.txt

