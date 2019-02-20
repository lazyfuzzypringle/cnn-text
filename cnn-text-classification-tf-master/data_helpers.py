import numpy as np
import re
import pickle as pkl
from gensim.models import Word2Vec
import os

UNK_TAG = 1
PAD_TAG = 0
UNK_TOKEN = 'UNK'
PAD_TOKEN = 'PAD'

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[1] for _ in positive_examples]
    negative_labels = [[0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

# x: list of input tokenized words
# vocab: dictionary
# output: list of indexed sentences
def sentence_indexing(x, vocab):
    indexed_x = []
    for sentence in x:
        index = [vocab.get(s, 'UNK') for s in sentence]
        indexed_x.append(index)
    return indexed_x

# pad indices into same length
# convert into array
def pad_indexed_sentence(x, maxlen):
    pad_x = []
    for sentence in x:
        padded = [sentence[:maxlen] if maxlen < len(sentence) else sentence+ [PAD_TAG]* (maxlen-len(sentence))]
        padded1 = sentence+[PAD_TAG]*(maxlen-len(sentence))
        pad_x.append(padded1)

    pad_x = np.array(pad_x, dtype=np.int64)
    return pad_x

# save gensim model as pkl. more complete than model.save
def train_word2vect(data, model_path, embedding_dim):
    if os.path.exists(model_path):
        with open(model_path,'rb') as f:
            model = pkl.load(f)
    else:
        model = Word2Vec(data, size=embedding_dim, window=5, min_count=1)
        with open(model_path,'wb') as f:
            pkl.dump(model, f)
    return model.wv, model.wv.vocab

# add pad tokens to vocab and embeddings
# vocab: type: gensim model.wv
# word_vects: pre-trained embedding, array type
def process_embeddings(word_vects, vocab, embedding_dim):
    word_embedding = None
    for w in list(vocab):
        if word_embedding is not None:
            word_embedding = np.concatenate((word_embedding, np.expand_dims(word_vects[w], axis=0)), axis=0)
        else:
            word_embedding = np.expand_dims(word_vects[w], 0)
    vocab = [PAD_TOKEN, UNK_TOKEN] + list(vocab)
    idx2word = {i: w for i, w in enumerate(vocab)}
    word2idx = {w: i for i, w in enumerate(vocab)}
    pad_vects = np.random.randn(2, embedding_dim)
    word_embedding = np.concatenate((pad_vects, word_embedding))
    return idx2word, word2idx, word_embedding

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
