from __future__ import print_function
from bs4 import BeautifulSoup
from nltk.stem.snowball import SnowballStemmer
from os import listdir
from os.path import isfile, join
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.externals import joblib

import codecs
import mpld3
import numpy as np
import nltk
import os
import pandas as pd
import re
import time

stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer("english")
dirPath = 'Dataset'

# functions

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]

    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)

    # remove token if it's a stopword
    tokens_without_stopwords = [token for token in filtered_tokens if (token not in stopwords)]
    
    stems = [stemmer.stem(t) for t in tokens_without_stopwords]

    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]

    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)

    # remove token if it's a stopword
    tokens_without_stopwords = [token for token in filtered_tokens if (token not in stopwords)]
    
    return tokens_without_stopwords

# Load documents

files = [f for f in listdir(dirPath) if isfile(join(dirPath, f))]

fileContents = []
for file in files:
    fileContents.append(open(dirPath + '/' + file).read())

# Tokenize and stemming

totalvocab_stemmed = []
totalvocab_tokenized = []
for i in fileContents:
    allwords_stemmed = tokenize_and_stem(i)
    totalvocab_stemmed.extend(allwords_stemmed)

    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)

vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)

# TF IDF Vectorizer

tfidf_vectorizer = TfidfVectorizer(max_df = 0.8, max_features = 200000,
    min_df = 0.15, stop_words = 'english',
    use_idf = True, tokenizer = tokenize_and_stem, ngram_range = (1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(fileContents)

terms = tfidf_vectorizer.get_feature_names()

# KMeans Clustering
start_time = time.time()

num_clusters = 4
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)

clusters = km.labels_.tolist()

# joblib.dump(km,  'doc_cluster.pkl')
# km = joblib.load('doc_cluster.pkl')
clusters = km.labels_.tolist()

documents = { 'doc_num': files, 'text': fileContents, 'cluster': clusters }

frame = pd.DataFrame(documents, index = [clusters] , columns = ['doc_num', 'cluster'])
end_time = time.time() - start_time

f1 = open('kmeans result.txt', 'w')
f1.write("Time collapsed: " + str(end_time) + "s\n")
f1.write("Top terms per cluster:\n")
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
for i in range(num_clusters):
    f1.write("Cluster[" + str(i) + "] words:")
    for ind in order_centroids[i, :11]:
        f1.write(" " + vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0],)
    f1.write("\n")
    f1.write("Cluster[" + str(i) + "] documents: ")
    for num in frame.ix[i]['doc_num'].values.tolist():
        f1.write(" " + num)
    f1.write("\n\n")