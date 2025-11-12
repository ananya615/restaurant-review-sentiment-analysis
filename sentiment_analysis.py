import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nltk.download('stopwords')

# Load dataset
df = pd.read_csv('/content/Restaurant reviews.csv')
df = df.drop(columns=["Restaurant", "Reviewer", "Metadata", "Time", "Pictures"], errors='ignore')

# Convert Rating to numeric
df["Rating"] = pd.to_numeric(df["Rating"], errors='coerce')


y = df["Rating"].replace({'Like': 3}).fillna(df["Rating"].median()).round().astype(int)
y = y.apply(lambda x: 1 if x >= 3 else 0)
X = df["Review"].astype(str)

# Text preprocessing
ps = PorterStemmer()
corpus = []
for review in X:
    review = re.sub('[^a-zA-Z]', ' ', review).lower()
    review = [ps.stem(word) for word in review.split() if word not in stopwords.words('english')]
    corpus.append(' '.join(review))

analyzer = SentimentIntensityAnalyzer()
sentiment_scores = [analyzer.polarity_scores(text)['compound'] + 1 for text in X]  # Shift to non-negative range
text_lengths = [len(text.split()) for text in X]


cv = CountVectorizer(ngram_range=(1,2), max_features=9000)
X_cv = cv.fit_transform(corpus).toarray()
tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=9000)
X_tfidf = tfidf.fit_transform(corpus).toarray()

X_cv = np.hstack((X_cv, np.array(sentiment_scores).reshape(-1,1), np.array(text_lengths).reshape(-1,1)))
X_tfidf = np.hstack((X_tfidf, np.array(sentiment_scores).reshape(-1,1), np.array(text_lengths).reshape(-1,1)))

#train-test split
X_train_cv, X_test_cv, y_train, y_test = train_test_split(X_cv, y, test_size=0.25, random_state=42)
X_train_tfidf, X_test_tfidf, _, _ = train_test_split(X_tfidf, y, test_size=0.25, random_state=42)

# Models
nb_model = MultinomialNB()
nb_model.fit(X_train_cv, y_train)

lr_model = LogisticRegression(max_iter=500)
lr_model.fit(X_train_tfidf, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_tfidf, y_train)

# Hybrid Models
stacking_clf = StackingClassifier(
    estimators=[('nb', nb_model), ('lr', lr_model)],
    final_estimator=LogisticRegression()
)
stacking_clf.fit(X_train_tfidf, y_train)

voting_clf = VotingClassifier(
    estimators=[('nb', nb_model), ('lr', lr_model), ('rf', rf_model)], voting='soft'
)
voting_clf.fit(X_train_tfidf, y_train)


y_pred_nb = nb_model.predict(X_test_cv)
y_pred_lr = lr_model.predict(X_test_tfidf)
y_pred_rf = rf_model.predict(X_test_tfidf)
y_pred_stack = stacking_clf.predict(X_test_tfidf)
y_pred_vote = voting_clf.predict(X_test_tfidf)

def evaluate_model(name, y_true, y_pred):
    print(f"\n{name} Model Performance:")
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))

evaluate_model("Naive Bayes", y_test, y_pred_nb)
evaluate_model("Logistic Regression", y_test, y_pred_lr)
evaluate_model("Random Forest", y_test, y_pred_rf)
evaluate_model("Stacking Classifier", y_test, y_pred_stack)
evaluate_model("Voting Classifier", y_test, y_pred_vote)
