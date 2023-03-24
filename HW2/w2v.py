from gensim.models import Word2Vec
import utils
from constants import *

# Function to train word2vec model
def make_word2vec_model(data,sg=1, min_count=1, size=EMBEDDING_SIZE, workers=2, window=3):
    data = list(data)
    data.append(['pad'])
    word2vec_file = './models/' + 'word2vec_' + str(size) + '_PAD.model'
    w2v_model = Word2Vec(data, min_count = min_count, vector_size = size, workers = workers, window = window, sg = sg)
    w2v_model.save(word2vec_file)

if __name__ == "__main__":
    # make_word2vec_model(utils.load_data()[0])
    pass