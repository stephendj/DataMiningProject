from __future__ import print_function
from bs4 import BeautifulSoup
from nltk.stem.snowball import SnowballStemmer
from os import listdir
from os.path import isfile, join
from scipy.cluster.hierarchy import ward, dendrogram
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.externals import joblib

import codecs
import matplotlib.pyplot as plt
import mpld3
import numpy as np
import nltk
import os
import pandas as pd
import re

stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer("english")
dirPath = 'Dataset1TuBes'

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

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
    min_df=0.15, stop_words='english',
    use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(fileContents)

terms = tfidf_vectorizer.get_feature_names()
dist = 1 - cosine_similarity(tfidf_matrix)

# Hierarchical Document Clustering

linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances

fig, ax = plt.subplots(figsize=(15, 20)) # set size
ax = dendrogram(linkage_matrix, orientation="right", labels=files);

plt.tick_params(\
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.tight_layout() #show plot with tight layout

#uncomment below to save figure
plt.savefig('ward_clusters.png', dpi=300) #save figure as ward_clusters