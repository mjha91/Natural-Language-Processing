import os
from pathlib import Path
import numpy
 
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
 
import string
from sklearn.feature_extraction.text import CountVectorizer 
 
import gensim
from gensim import corpora
 
import pandas as pd
 
######################## user defined
dir = os.path.join(os.path.dirname(__file__),"textfile_directory")
metadata_file = "metadata.csv"
result_file = "result.txt"
ngram_length = 4
k = 5            # number of topics
p = 50           # number of passes - number of times you want to go through the entire corpus
iter = 1000      # number of iterations
########################
 
# part of alpha in lda
    # theta: per-document topic distributions; with a higher theta, documents are made up of more topics
    #lambda: per-topic word distribution; with a high lambda, topics are made up of most of the words in the corpus
 
stop = set(stopwords.words('english'))  # remove stop words such as: the, is, are
punc = set(string.punctuation)  # remove puntuation
lemma = WordNetLemmatizer()     # stemming can often create non-existent words, whereas lemmas are actual words
                                # lemmatize takes a part of speech parameter, "pos" If not supplied, the default is "noun"
vectorizer = CountVectorizer(ngram_range = (1, ngram_length)).build_analyzer()
 
def create_ngram(text_file):
    content = Path(text_file).read_text()
    content  = content.replace(".","\n")    # breaking paragraphs into sentences by splitting at . , ; :
    content  = content.replace(",","\n")
    content  = content.replace(";","\n")
    content  = content.replace(":","\n")
 
    ngrams = []
    for line in content.split("\n"):
        stop_free = " ".join([i for i in line.lower().split() if i not in stop])
        punc_free = ''.join(ch for ch in stop_free if ch not in punc)
        normalized_noun = " ".join(lemma.lemmatize(word) for word in punc_free.split()) # lemmatize noun: cars to car
        normalized = " ".join(lemma.lemmatize(word, pos = "v") for word in normalized_noun.split()) # lemmatize verb: reduced to reduce
        ngrams = ngrams + vectorizer(normalized)
 
    return ngrams
 
# get metadata like date, time, s&p and fed futures movement
metadata = numpy.genfromtxt( metadata_file, dtype = None, delimiter = ",")
 
# create ngram for each text file in metadata
ngram_array = [create_ngram(os.path.join(dir, row[0].decode("utf-8"))) for row in metadata] 
 
# create Dictionary and Document Term Matrix
dictionary = corpora.Dictionary(ngram_array)
doc_term_matrix = [dictionary.doc2bow(ngrams) for ngrams in ngram_array]
 
# Creating the object for HDP model using gensim library
hdp = gensim.models.hdpmodel.HdpModel
hdpmodel = hdp(doc_term_matrix, id2word = dictionary)
hdpmodel.optimal_ordering() #ordersntopics based on probability of occuring
 
# hdp model by default creates 150 topics, this function helps in getting proability of each topic
def topic_prob_extractor(gensim_hdp):
    shown_topics = gensim_hdp.show_topics(num_topics=-1, formatted=False)
    topics_nos = [x[0] for x in shown_topics ]  # number from 0 to 149
    weights = [ sum([item[1] for item in shown_topics[topicN][1]]) for topicN in topics_nos ] # item[1] gives you weight of the word, so it is adding all the weight
 
    df = pd.DataFrame({'topic_id' : topics_nos, 'weight' : weights})
    df.to_csv("topic_weight.csv", sep='\t')
    return df
 
topic_prob_extractor(hdpmodel)
 
#print(hdpmodel.get_topics())
 
#print(hdpmodel.hdp_to_lda())
print(hdpmodel.suggested_lda_model())
