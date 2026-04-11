import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
# from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
# from imblearn.over_sampling import SMOTE
from sklearn.pipeline import make_pipeline
import joblib

data = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/refs/heads/master/data/sms.tsv"
df = pd.read_csv(data, sep='\t',header=None, names=['label', 'msg'])

x=df['msg']
y=df['label']
encoder = LabelEncoder()
y = encoder.fit_transform(y)
# print(y.sample(5))
# x_tfidf = tfidfvect.fit_transform(x)

model = make_pipeline(
TfidfVectorizer(stop_words="english", max_df=0.9, min_df=2, lowercase=True  ),
LogisticRegression(max_iter=1000, class_weight='balanced')
)

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2, stratify=y )

# smote = SMOTE()
# X_train, y_train = smote.fit_resample(X_train, y_train)
# mnmodel = MultinomialNB()
# mnmodel.fit(X_train, y_train)
# y_pred = mnmodel.predict(X_test)
# accracy = accuracy_score(y_test, y_pred)
# classification_rep = classification_report(y_test, y_pred)
# print("{} {}".format(accracy, classification_rep))


model.fit(X_train, y_train)
y_logpred=model.predict(X_test)

y_proba = model.predict_proba(X_test)[:,1]
logacc = accuracy_score(y_test,y_logpred )
confmatrix = classification_report(y_test, y_logpred)

print("{} {}".format( logacc, confmatrix))

joblib.dump(model, 'spam_detector.pkl')
