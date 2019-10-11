import re
import numpy as np
from glob import glob

def GetAllVocab(filename):
    with open(filename, "r") as f:
        all_lines = f.readlines()
        all_words = []
        for word in all_lines:
            words = word.split()
            all_words.append(words[0])
    return all_words

data_vocab_file = "aclImdb/imdb.vocab"
vector_vocab_file = "vocab.txt"

data_vocab = GetAllVocab(data_vocab_file)
vector_vocab = GetAllVocab(vector_vocab_file)

count = 0
for word in data_vocab:
    if word not in vector_vocab:
        count += 1

print(count)
