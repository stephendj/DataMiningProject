from collections import defaultdict
from gensim import corpora, models
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from os import listdir
from os.path import isfile, join
from stop_words import get_stop_words

import gensim
import time

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
    
# load documents
dirPath = 'Dataset'
files = [f for f in listdir(dirPath) if isfile(join(dirPath, f))]

fileContents = []
for file in files:
    fileContents.append(open(dirPath + '/' + file).read())

# list for tokenized documents in loop
texts = []

# loop through document list
for i in fileContents:
    
    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]
    
    # stem tokens
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    
    # add tokens to list
    texts.append(stemmed_tokens)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)
    
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

# generate LDA model
start_time = time.time()
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 4, id2word = dictionary, passes = 20)
end_time = time.time() - start_time

f1 = open('LDA result.txt', 'w')
f1.write("Time collapsed: " + str(end_time) + "s\n")

for i in ldamodel.show_topics():
    f1.write("Cluster[" + str(i[0]) + "] words: ")
    f1.write(str(i[1]) + " ")
    f1.write("\n")
	
results = {}
for i in range(len(files)):
    for index, score in sorted(ldamodel[corpus[i]], key = lambda tup: -1 * tup[1]):
        results[files[i]] = index
        break

v = defaultdict(list)

for key, value in results.items():
    v[value].append(key)

for key, value in v.items():
    f1.write("\n")
    f1.write("Cluster[" + str(key) + "] documents: ")
    f1.write(str(value) + " ")
    f1.write("\n")