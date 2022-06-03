from gensim.models import Word2Vec
import numpy as np

sentences = np.load('data/words_CleanNewsFilter.pkl', allow_pickle=True)
print('data loaded')

model = Word2Vec(sentences=sentences, vector_size=300, window=5, min_count=5, workers=12)
model.save("out/word2vec.model")

