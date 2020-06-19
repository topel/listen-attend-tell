
from numpy import load as np_load
import pickle


# import socket
# host_name = socket.gethostname()
# print(host_name)

from pynlpl.lm import lm

from clotho_dataloader.data_handling.my_clotho_data_loader import create_dictionaries

import glob

__author__ = "Thomas Pellegrini - 2020"

data_dir='../clotho-dataset/data/'
WORD_LIST = pickle.load(open(data_dir + "/words_list.p", "rb"))# 4367 word types
WORD_FREQ = pickle.load(open(data_dir + "/words_frequencies.p", "rb"))

word2index, index2word = create_dictionaries(WORD_LIST)

def gather_captions_to_text(caption_dir, out_fpath):

    fh = open(out_fpath, 'wt')
    i = 0
    for npy_fpath in glob.glob(caption_dir + '/*.npy'):

        recarray = np_load(str(npy_fpath), allow_pickle=True)
        word_indices_list = recarray['words_ind'][0]
        # print(word_indices_list)
        word_str_list = [index2word[w] for w in word_indices_list]
        # word_str = ' '.join(word_str_list).replace('<sos> ', '')
        word_str = ' '.join(word_str_list)
        # print(npy_fpath, word_str)
        fh.write(word_str + '\n')
        i += 1
        # if i==2: break
    print("wrote %d lines to file"%i)

# subset = 'clotho_dataset_dev'
subset = 'clotho_dataset_eva'
caption_dir=data_dir + subset

# out_fpath = '../clotho-dataset/lm/dev.txt'
out_fpath = '../clotho-dataset/lm/eva.txt'
# gather_captions_to_text(caption_dir, out_fpath)

# building  3-g LM
# https://cmusphinx.github.io/wiki/tutoriallm/#training-an-arpa-model-with-srilm

# $ ~/tools/kaldi/tools/srilm/lm/bin/i686-m64/ngram-count -kndiscount -interpolate -text ../clotho-dataset/lm/dev.txt -lm ../clotho-dataset/lm/dev.lm
# $ ~/tools/kaldi/tools/srilm/lm/bin/i686-m64/ngram -lm ../clotho-dataset/lm/dev.lm -ppl ../clotho-dataset/lm/eva.txt
# file ../clotho-dataset/lm/eva.txt: 5225 sentences, 64350 words, 0 OOVs
# 0 zeroprobs, logprob= -114470.2 ppl= 44.18532 ppl1= 60.09924

# $ ~/tools/kaldi/tools/srilm/lm/bin/i686-m64/ngram -lm ../clotho-dataset/lm/dev.lm -prune 1e-8 -write-lm ../clotho-dataset/lm/dev_pruned.lm
# $ ~/tools/kaldi/tools/srilm/lm/bin/i686-m64/ngram -lm ../clotho-dataset/lm/dev_pruned.lm -ppl ../clotho-dataset/lm/eva.txt
# file ../clotho-dataset/lm/eva.txt: 5225 sentences, 64350 words, 0 OOVs
# 0 zeroprobs, logprob= -114658.9 ppl= 44.46208 ppl1= 60.50634

# lm_path=b'../clotho-dataset/lm/dev.lm'
#
# lm = LM(lm_path, lower=False)
#
#
# print(len([b"man", b"a"]))
# print(lm.logprob_strings(lm, b"is", [b"man", b"a"]))

lm = lm.ARPALanguageModel('../clotho-dataset/lm/dev.lm')

print("man", lm.scoreword("man"))

print("a man", lm.scoreword("man", history=("a",)))

print("a hyman", lm.scoreword("hyman", history=("a",)))
