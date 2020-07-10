"""Provides functions to train a seq2seq model on clotho, either from scratch or from a checkpoint"""

import torch
# import torch.nn as nn
import torch.optim as optim

from models import Seq2Seq, masked_ce_loss, count_parameters

import numpy as np
import random

from utils_train_val_test import train, val
from utils import save_checkpoint, get_params_dict

from clotho_dataloader.data_handling.my_clotho_data_loader import get_clotho_loader, create_dictionaries, modify_vocab

import pickle
from pathlib import Path
import matplotlib.pyplot as plt

import sys

import socket

host_name = socket.gethostname()
print(host_name)

__author__ = "Thomas Pellegrini - 2020"

data_dir = '../clotho-dataset/data'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True

data_dir_path=Path(data_dir)
captions_gt_fpath = data_dir + "/clotho_captions_evaluation.csv"

LETTER_LIST = pickle.load(open(data_dir + "/characters_list.p", "rb"))
LETTER_FREQ = pickle.load(open(data_dir + "/characters_frequencies.p", "rb"))

WORD_LIST = pickle.load(open(data_dir + "/words_list.p", "rb"))# 4367 word types
WORD_FREQ = pickle.load(open(data_dir + "/words_frequencies.p", "rb"))

# WORD_COUNT_THRESHOLD = 10
WORD_COUNT_THRESHOLD = None


word2index, index2word = create_dictionaries(LETTER_LIST)
word2index, index2word = create_dictionaries(WORD_LIST)

if WORD_COUNT_THRESHOLD is not None:
    print("WORD_COUNT_THRESHOLD =", WORD_COUNT_THRESHOLD)
    word2index, index2word, WORD_LIST, mapping_index_dict = modify_vocab(WORD_LIST, WORD_FREQ, WORD_COUNT_THRESHOLD)
else:
    mapping_index_dict = None
print("Vocab:", len(WORD_LIST) )

def main():

    # Train?
    do_pretrain_decoder_as_an_lm = False
    # weight_decay=0.
    weight_decay = 1e-6
    do_load_pretrained_embeddings = False
    freeze_embeddings = False

    # Load pre-trained?
    do_load_checkpoint = False
    do_load_decoder = False

    checkpoint_pathname = 'checkpoints/40_1.6245147620307074_2.6875626488429742_checkpoint.tar'
    save_dir='checkpoints'

    input_dim = 64
    vocab_size = len(WORD_LIST) # seq2seq at WORD LEVEL

    if do_load_pretrained_embeddings:
        emb_fpath='/tmpdir/pellegri/corpus/clotho-dataset/lm/word2vec_dev_128.pth'
    else:
        emb_fpath = None

    use_spec_augment = False
    use_gumbel_noise = False

    encoder_hidden_dim = 128
    embedding_dim = 128
    value_size, key_size, query_size = [64] * 3  # these could be different from embedding_dim


    # teacher_forcing_ratio = 1.
    teacher_forcing_ratio = float(sys.argv[1]) # 0.98

    # pBLSTM_time_reductions = [2, 2, 2]
    config_pBLSTM_str = sys.argv[2:]
    pBLSTM_time_reductions = [int(config_pBLSTM_str[i]) for i in range(len(config_pBLSTM_str))]
    print("config pBLSTM", pBLSTM_time_reductions)
    # nb_pBLSTM_layers = len(pBLSTM_time_reductions) # from 1 to 3
    # [2,2] 0 --> 2887375 params
    # [2,2] 8 --> 2904015 params

    decoder_hidden_size_1 = 128
    decoder_hidden_size_2 = 64

    print("use Gumbel noise", use_gumbel_noise)
    print("use teacher forcing", teacher_forcing_ratio)
    print("use SpecAugment", use_spec_augment)

    model = Seq2Seq(input_dim=input_dim, vocab_size=vocab_size, encoder_hidden_dim=encoder_hidden_dim,
                        use_spec_augment=use_spec_augment,
                        embedding_dim=embedding_dim,
                        decoder_hidden_size_1=decoder_hidden_size_1,
                        decoder_hidden_size_2=decoder_hidden_size_2, query_size=query_size,
                        value_size=value_size, key_size=key_size, isAttended=True,
                        pBLSTM_time_reductions=pBLSTM_time_reductions,
                        emb_fpath=emb_fpath, freeze_embeddings=freeze_embeddings,
                        teacher_forcing_ratio=teacher_forcing_ratio, # beam_size=beam_size, lm_weight=lm_weight,
                        word2index=word2index, return_attention_masks=False, device=DEVICE)

    print(model)

    num_params = count_parameters(model)
    print("num trainable params:", num_params)


    if do_load_checkpoint:
        print("Loading checkpoint: ", checkpoint_pathname)
        model_checkpoint = torch.load(
            checkpoint_pathname, map_location=DEVICE
        )
        model_state = model_checkpoint["model"]
        model.load_state_dict(model_state)
        model = model.to(DEVICE)

        lr = 0.001
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer_state = model_checkpoint["model_optim"]
        optimizer.load_state_dict(optimizer_state)
        start_train_epoch = model_checkpoint["iteration"]

    elif do_pretrain_decoder_as_an_lm:
        # we train the decoder weights only
        lr = 0.001
        optimizer = optim.Adam(model.decoder.parameters(), lr=lr, weight_decay=weight_decay)
        start_train_epoch = 0

    elif do_load_decoder:
        print("Loading decoder checkpoint: ", checkpoint_pathname)
        decoder_checkpoint = torch.load(
            checkpoint_pathname, map_location=DEVICE
        )
        decoder_state = decoder_checkpoint["model"]
        model.decoder.load_state_dict(decoder_state)
        model = model.to(DEVICE)

        lr = 0.001
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer_state = decoder_checkpoint["model_optim"]
        optimizer.load_state_dict(optimizer_state)
        start_train_epoch = 0 # decoder_checkpoint["iteration"]
    else:
        lr = 0.0005
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        start_train_epoch = 0

    model = model.to(DEVICE)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20], gamma=10.)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=.1)

    # criterion = nn.CrossEntropyLoss(reduction=None)
    criterion = masked_ce_loss
    # nepochs = 25
    nepochs = start_train_epoch + 60
    save_every = 5

    batch_size = 64
    val_batch_size = 100

    print("nepochs", nepochs)
    print("batch_size", batch_size)
    print("val_batch_size", val_batch_size)
    print("learning rate", lr)

    model_name='seq2seq'
    corpus_name='clotho'
    params_dict = get_params_dict(model_name, corpus_name, input_dim, vocab_size, embedding_dim, value_size,
                                  pBLSTM_time_reductions, teacher_forcing_ratio, use_gumbel_noise, use_spec_augment,
                                  lr, weight_decay, emb_fpath, freeze_embeddings)

    # training data loader
    split = 'clotho_dataset_dev'
    input_field_name = 'features'
    # output_field_name = 'caption'
    # output_field_name = 'caption_ind'
    output_field_name = 'words_ind'
    # output_field_name = 'chars_ind'
    fileid_field_name = 'file_name'

    # load the whole subset in memory
    load_into_memory = False

    nb_t_steps_pad = 'max'
    has_gt_text = True
    shuffle = True
    drop_last = True
    # input_pad_at='start'
    input_pad_at = 'end'
    output_pad_at = 'end'
    num_workers = 0

    print(" training subset:", split)
    train_loader = get_clotho_loader(data_dir=data_dir_path,
                                            split=split,
                                            input_field_name=input_field_name,
                                            output_field_name=output_field_name,
                                            fileid_field_name=fileid_field_name,
                                            load_into_memory=load_into_memory,
                                            batch_size=batch_size,
                                            nb_t_steps_pad=nb_t_steps_pad,  #: Union[AnyStr, Tuple[int, int]],
                                            has_gt_text=has_gt_text,
                                            shuffle=shuffle,
                                            drop_last=drop_last,
                                            input_pad_at=input_pad_at,
                                            output_pad_at=output_pad_at,
                                            mapping_index_dict=mapping_index_dict,
                                            num_workers=num_workers)

    # validation data loader
    split = 'clotho_dataset_eva'
    # split = 'clotho_dataset_eva_50'
    shuffle = False
    drop_last = False
    val_loader = get_clotho_loader(data_dir=data_dir_path,
                                     split=split,
                                     input_field_name=input_field_name,
                                     output_field_name=output_field_name,
                                     fileid_field_name=fileid_field_name,
                                     load_into_memory=load_into_memory,
                                     batch_size=val_batch_size,
                                     nb_t_steps_pad=nb_t_steps_pad,  #: Union[AnyStr, Tuple[int, int]],
                                     has_gt_text=has_gt_text,
                                     shuffle=shuffle,
                                     drop_last=drop_last,
                                     input_pad_at=input_pad_at,
                                     output_pad_at=output_pad_at,
                                     mapping_index_dict=mapping_index_dict,
                                     num_workers=num_workers)

    # for i, test_batch in enumerate(test_loader):
    #     speech_batch, speech_lengths, ids_batch  = test_batch
    #     print(speech_batch.size(), speech_lengths, ids_batch)
    #     #     print(text_batch)
    #     #     print(text_lengths)
    #     if i == 10: break
    # return

    train_losses, val_losses = [], []
    print("Begin training...")
    for epoch in range(start_train_epoch, nepochs):

        log_fh = open('train.log', mode='at')

        print("epoch:", epoch)
        print("train subset...")
        train_loss = train(model,
                           train_loader,
                           criterion,
                           optimizer,
                           epoch,
                           pretrain_decoder=do_pretrain_decoder_as_an_lm,
                           use_gumbel_noise=use_gumbel_noise,
                           device=DEVICE)
        train_losses.append(train_loss)

        print("val subset...")
        val_loss = val(model,
                       val_loader,
                       criterion,
                       epoch,
                       pretrain_decoder=do_pretrain_decoder_as_an_lm,
                       use_gumbel_noise=False,
                       print_captions=True,
                       index2word=index2word,
                       device=DEVICE)
        val_losses.append(val_loss)

        log_fh.write("  epoch %d:\ttrain_loss: %.5f\tval_loss: %.5f\n" % (epoch, train_loss, val_loss))
        if epoch == nepochs-1: log_fh.write("val_loss: %.5f\n" %val_loss)
        log_fh.close()

        if epoch>start_train_epoch and (epoch+1) % save_every == 0:
            model_dir = save_checkpoint(save_dir,
                                        model,
                                        optimizer,
                                        epoch+1,
                                        train_loss,
                                        val_loss,
                                        params_dict,
                                        do_pretrain_decoder_as_an_lm)
        scheduler.step()

    plt.plot(train_losses, 'k', label='train')
    plt.plot(val_losses, 'b', label='val')
    plt.legend()
    plt.savefig(model_dir + "/loss.png")



if __name__ == '__main__':
    main()

