import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.porter import PorterStemmer
from constants import *

def clean_data():
    # read the entire file into a python array
    data=[]
    cnt=0
    with open(PATH_TO_YELP_REVIEWS, 'r', encoding='utf-8') as f:
        while True:
            if(cnt==100000):
                break
            line = f.readline()
            # remove the trailing "\n" from each line
            line=line.rstrip()
            data.append(line)
            cnt+=1

    data = "[" + ','.join(data) + "]"

    # now, load it into pandas
    data = pd.read_json(data)

    # Function to map stars to sentiment
    def map_sentiment(stars_received):
        if stars_received <= 2:
            return 0
        elif stars_received == 3:
            return 1
        else:
            return 2
    # Mapping stars to sentiment into three categories
    data['sentiment'] = [ map_sentiment(x) for x in data['stars']]

    data = pd.concat([data[data['sentiment'] == 1].head(10000), data[data['sentiment'] == -1].head(10000), data[data['sentiment'] == 0].head(10000)])

    # Tokenize the text column to get the new column 'tokenized_text'
    data['tokenized_text'] = [simple_preprocess(line, deacc=True) for line in data['text']] 
    porter_stemmer = PorterStemmer()
    # Get the stemmed_tokens
    data['stemmed_tokens'] = [[porter_stemmer.stem(word) for word in tokens] for tokens in data['tokenized_text'] ]

    data=data[['stemmed_tokens','sentiment']]

    data.to_csv(PATH_TO_CLEAN_DATA, index=False)

def load_data():
    df=pd.read_csv(PATH_TO_CLEAN_DATA,converters={'stemmed_tokens': lambda x: list(map(lambda s: s.strip("'"),x.strip("[]").split(", ")))})
    return df['stemmed_tokens'].values,df['sentiment'].values

def preproc_X(X):
    X=list(X)
    list_len = [len(i) for i in X]
    max_sen_len=max(list_len)
    w2vmodel = gensim.models.KeyedVectors.load('./models/' + 'word2vec_'+str(EMBEDDING_SIZE)+'_PAD.model')
    padding_idx = w2vmodel.wv.key_to_index['pad']
    out=[]
    for x in X:
        padded_X = [padding_idx for i in range(max_sen_len)]
        i = 0
        for word in x:
            if word not in w2vmodel.wv.key_to_index:
                padded_X[i] = 0
            else:
                padded_X[i] = w2vmodel.wv.key_to_index[word]
            i += 1
        out.append(padded_X)
    return out

def preproc_y(y):
    return list(y)

def preproc_data(X,y):
    return preproc_X(X),preproc_y(y)

if __name__ == "__main__":
    # clean_data()
    pass