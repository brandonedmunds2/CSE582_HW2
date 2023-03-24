INPUT_FOLDER='./unproc_data/'
CLEAN_FOLDER='./data/'

PATH_TO_YELP_REVIEWS = INPUT_FOLDER + 'review.json'
PATH_TO_CLEAN_DATA = CLEAN_FOLDER + 'data.csv'

EMBEDDING_SIZE=25
HIDDEN_SIZE=32
NUM_FILTERS = 12
NUM_CLASSES = 3
EPOCHS=10
LR=0.01
TRAIN_BATCH_SIZE=32
TEST_BATCH_SIZE=1024
MAX_REVIEW_SIZE=927

if __name__ == '__main__':
    pass