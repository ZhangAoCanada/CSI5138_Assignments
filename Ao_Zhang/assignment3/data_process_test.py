import re
import numpy as np
from glob import glob


def cleanSentences(string):
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

train_pos_files = glob("aclImdb/train/pos/*.txt")
train_neg_files = glob("aclImdb/train/neg/*.txt")

test_pos_files = glob("aclImdb/test/pos/*.txt")
test_neg_files = glob("aclImdb/test/neg/*.txt")

all_txt = None

for each_file in train_pos_files:
    f = open(each_file, "r")
    line=f.readline()
    cleaned_line = cleanSentences(line)
    if all_txt is None:
        all_txt = cleaned_line
    else:
        all_txt += " "
        all_txt += cleaned_line

for each_file in train_neg_files:
    f = open(each_file, "r")
    line=f.readline()
    cleaned_line = cleanSentences(line)
    if all_txt is None:
        all_txt = cleaned_line
    else:
        all_txt += " "
        all_txt += cleaned_line

for each_file in test_pos_files:
    f = open(each_file, "r")
    line=f.readline()
    cleaned_line = cleanSentences(line)
    if all_txt is None:
        all_txt = cleaned_line
    else:
        all_txt += "\n"
        all_txt += cleaned_line

for each_file in test_neg_files:
    f = open(each_file, "r")
    line=f.readline()
    cleaned_line = cleanSentences(line)
    if all_txt is None:
        all_txt = cleaned_line
    else:
        all_txt += " "
        all_txt += cleaned_line


with open("aclimdb_data.txt", "w") as f:
    f.write(all_txt)