# user defined
dir = 'abc' #contains text files
w2vdimension = 300 # The quality for vector representations improves as you increase the vector size till 300 dimensions
              # default is 100
inputwords = ['work', 'play']
 
# import
from gensim.models import Word2Vec
 
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
 
from matplotlib import pyplot
import numpy as np
import os
 
# define training data: we need list of list as input
sentences = []
txtfiles = [os.path.join(dir,t) for t in os.listdir(dir)]
for txtfile in txtfiles: 
 
    #read, make lowercase
    textfile = open(txtfile, 'r')
    text = textfile.read().lower()
    textfile.close()
    lines = text.split('\n')
 
    for line in lines:
        linelist = line.split()
        sentences.append(linelist)
 
# train model on sentences; we ignore words which occurs less than min_count
model = Word2Vec(sentences, size = w2vdimension, min_count = 100)
print('trained')
model.save('w2v')
 
# load
model = Word2Vec.load('w2v')
print('saved and loaded')
 
# fit a 2d PCA model to the vectors
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
 
# create a scatter plot of the projection for all
pyplot.figure(figsize=(160, 160))
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
 
pyplot.savefig('word2vec.png')
pyplot.show()
 
# create projection for a small area# create projection for a small area
def display_closestwords_tsnescatterplot(model, word):
 
    arr = np.empty((0,w2vdimension), dtype='f') # w2vdimension, 100 is the default size
    word_labels = [word]
 
    # get close words
    close_words = model.similar_by_word(word)
 
    # add the vector for each of the closest words to the array
    arr = np.append(arr, np.array([model[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)
 
    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)
 
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    pyplot.scatter(x_coords, y_coords)
 
    for label, x, y in zip(word_labels, x_coords, y_coords):
        pyplot.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
 
    pyplot.xlim(x_coords.min() + 0.0001, x_coords.max() + 0.0001)
    pyplot.ylim(y_coords.min() + 0.0001, y_coords.max() + 0.0001)
    pyplot.savefig(word + 'w2v.png')
    pyplot.show()
 
# output for the words we are interested in
for w in inputwords:
    display_closestwords_tsnescatterplot(model, w)
    print(model.wv.similar_by_word(w, topn = 20, restrict_vocab=None))
