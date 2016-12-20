from gensim import corpora, models
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from os import listdir
from os.path import isfile, join
from stop_words import get_stop_words
import gensim

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
    
# load documents
dirPath = '12 Dataset @3'
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
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 2, id2word = dictionary, passes = 20)
print(ldamodel.print_topics(num_topics = 2, num_words = 3))