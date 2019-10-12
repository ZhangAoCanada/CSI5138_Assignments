import os
import re
import numpy as np
from glob import glob
from tqdm import tqdm

#####################################################################
# This class is for reading the vocabulary and word vectors saved
# in the current directory.
# 
# Then, save the word vector as numpy array in the directory
# ``dataset_numpy'' with vector indexes pointing to a vocabulary
# python dict.
#
# By doing that, we can easily search the word vector using function
# tf.nn.embedding_lookup()
#####################################################################
class WordVectorAndList:
    def __init__(self, vocab_file, vector_file):
        """
        Function:
            Initialization of all values. Note that when the wordvector
        numpy array exists, read the existing array instead of re-produce
        it again for saving running time.
        """
        self.vocab_file = vocab_file
        self.vector_file = vector_file
        self.vocab_list = self.VocabList(self.vocab_file)
        if os.path.exists("dataset_numpy/wordvector.npy"):
            self.word_vector = np.load("dataset_numpy/wordvector.npy")
        else:
            self.word_vector = self.WordVector(self.vector_file, self.vocab_list)

    def VocabList(self, filename):
        """
        Function:
            Build a vocabulary python dictionary with format as:
        {
            "word1" : index1
            "word2" : index2
            ...
         }
        """
        with open(filename, "r") as f:
            all_lines = f.readlines()
            vocab_dict = {}
            count = 0
            for line in all_lines:
                count += 1
                words = line.split()
                vocab_word = words[0]
                vocab_dict[vocab_word] = count
        return vocab_dict

    def WordVector(self, filename, vocab_list):
        """
        Funtion:
            Read the word vector trained by Glove and transfer it
        into np.array with first axies pointing to the word index.
        """
        with open(filename, "r") as f:
            all_lines = f.readlines()
            output_array = np.zeros((len(vocab_list) + 1, 50), dtype = np.float32)
            for line in all_lines:
                characters = line.split()
                if characters[0] not in vocab_list:
                    continue
                word_ind = vocab_list[characters[0]]
                word_vector = []
                for i in range(1, len(characters)):
                    word_vector.append(float(characters[i]))
                word_vector = np.array(word_vector)
                output_array[word_ind] = word_vector
        if not os.path.isdir("dataset_numpy"):
            os.mkdir("dataset_numpy")
        np.save("dataset_numpy/wordvector.npy", output_array)
        return output_array


# # vocabulary file and word vector file from Glove.
# vocab_file = "vocab.txt"
# vector_file = "vectors.txt"

# # get vocabulary and wordvector
# words_dict_creater = WordVectorAndList(vocab_file, vector_file)
# word_list = words_dict_creater.vocab_list
# word_vector = words_dict_creater.word_vector

# print(len(word_list))
# print(len(word_vector))


def FindAllSequanceLen(dataset_names, all_length):
    """
    Function:
        Scan the data set and append all file length into one list.
    """
    for file_name in dataset_names:
        with open(file_name, "r") as f:
            line = f.readline()
            words = line.split()
            all_length.append(len(words))
    return all_length

def PlotLenHist(train_pos_files, train_neg_files, test_pos_files, test_neg_files):
    """
    Function:
        Find all length w.r.t each dataset file, then plot the distribution
    of the length.
    """
    all_length = []
    all_length = FindAllSequanceLen(train_pos_files, all_length)
    all_length = FindAllSequanceLen(train_neg_files, all_length)
    all_length = FindAllSequanceLen(test_pos_files, all_length)
    all_length = FindAllSequanceLen(test_neg_files, all_length)
    all_length = np.array(all_length)
    print([all_length.min(), all_length.max()])

# all file names (Caution: dataset should be in the current dir)
train_pos_files = glob("aclImdb/train/pos/*.txt")
train_neg_files = glob("aclImdb/train/neg/*.txt")
test_pos_files = glob("aclImdb/test/pos/*.txt")
test_neg_files = glob("aclImdb/test/neg/*.txt")

PlotLenHist(train_pos_files, train_neg_files, test_pos_files, test_neg_files)
