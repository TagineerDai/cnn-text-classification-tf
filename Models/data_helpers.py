import numpy as np
import re, os, sys, pickle
import itertools
from collections import Counter


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


def load_data_and_labels(path):
    """
    Loads 12 newsgroup data from files, splits the data into words and append labels.
    Returns split sentences and labels.
    """
    themes = os.listdir(path)
    
    contents = []
    labels = []
    for idx, theme in enumerate(themes):
        # Label one hot[1, 0, 0, 0]
        label = [0 for _ in themes]
        label[idx] = 1
        word_list = []
        label_list = []
        f_list = os.listdir(os.path.join(path, theme))
        for f in f_list:
            try:
                lines = open(os.path.join(path, theme, f), 'r').readlines()
            except:
                lines = []
            words = [line.strip() for line in lines]
            words = [clean_str(word) for word in words]
            word_list.extend(words)
            label_list.append(label)
        labels.extend(label_list)
        contents.extend(word_list)
    return [contents, labels]

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
