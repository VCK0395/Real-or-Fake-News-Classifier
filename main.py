# Import the necessary modules
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv('fake_or_real_news.csv')

# Print the head of df
print(df.head())

# Create a series
X = df['text']
y = df.label

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=53)

# Initialize a CountVectorizer object
count_vectorizer = CountVectorizer(stop_words='english')

# Transform the training data and test data
count_train = count_vectorizer.fit_transform(X_train, y_train)

count_test = count_vectorizer.transform(X_test)

# Print the first 10 features of the count_vectorizer
print(count_vectorizer.get_feature_names()[:10])

# Initialize a TfidfVectorizer object
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Transform the training data and test data
tfidf_train = tfidf_vectorizer.fit_transform(X_train, y_test)

tfidf_test = tfidf_vectorizer.transform(X_test)

# Print the first 10 features
print(tfidf_vectorizer.get_feature_names()[:10])

# Print the first 5 vectors of the tfidf training data
print(tfidf_train.A[:5])

# Instantiate a Multinomial Naive Bayes classifier
nb_classifier = MultinomialNB()

# Create the model and predictions for countvector object
nb_classifier.fit(count_train, y_train)

pred = nb_classifier.predict(count_test)

# Calculate the accuracy score: score
score = metrics.accuracy_score(y_test, pred)
print(score)

# Calculate the confusion matrix
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
print(cm)

# Create a Multinomial Naive Bayes classifier
nb_classifier = MultinomialNB()

# Create the model and predictions for tfidf object
nb_classifier.fit(tfidf_train, y_train)

pred = nb_classifier.predict(tfidf_test)

# Calculate the accuracy score: score
score = metrics.accuracy_score(y_test, pred)
print(score)

# Calculate the confusion matrix
cm = metrics.confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
print(cm)

