""" Loads the sentiment analysis documents and labels

Sam Scott, Mohawk College, 2021

Modified by Nick Milanovic to use TfidVectorizer
"""

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
### Load docs and labels
filenames = ["amazon_cells_labelled.txt", "imdb_labelled.txt", "yelp_labelled.txt"]
docs = []
labels = []
for filename in filenames:
    with open("sentiment/"+filename) as file:
        for line in file:
            line = line.strip()
            labels.append(int(line[-1]))
            docs.append(line[:-2].strip())



## vectorize
vectorizer = TfidfVectorizer(ngram_range=(1,2),min_df=2)
vectorizer.fit(docs)
vectors = vectorizer.transform(docs)

## train classifier
clf = DecisionTreeClassifier()
clf.fit(vectors, labels)

## pickle
from joblib import dump
dump(clf, 'classifier.joblib')
dump(vectorizer, 'vectorizer.joblib')