"""Provides functions to do greedy decoding and beam search decoding on validation and test sets"""

import torch

from models import Seq2Seq, masked_ce_loss, masked_ce_loss_per_utt, count_parameters, BeamSeq2Seq

import numpy as np
import random

from utils_train_val_test import decode_val, bs_decode_val, decode_test, bs_decode_test, score_test_captions
from utils import get_params_dict, plot_att_masks_to_png_files, write_csv_prediction_file, load_gt_captions, read_csv_prediction_file, index2words

from clotho_dataloader.data_handling.my_clotho_data_loader import get_clotho_loader, create_dictionaries, modify_vocab
from eval_metrics import evaluate_metrics_from_lists

import pickle
from pathlib import Path

import sys

import socket
host_name = socket.gethostname()
print(host_name)

__author__ = "Thomas Pellegrini - 2020"

data_dir = '../clotho-dataset/data'
DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'

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

### If you want to use the words occurring at least ten times in dev, uncomment the following line
# WORD_COUNT_THRESHOLD = 10
WORD_COUNT_THRESHOLD = None
print("\n !!! WORD_COUNT_THRESHOLD = ", WORD_COUNT_THRESHOLD, " !!!\n")

letter2index, index2letter = create_dictionaries(LETTER_LIST)
word2index, index2word = create_dictionaries(WORD_LIST)

if WORD_COUNT_THRESHOLD is not None:
    print("WORD_COUNT_THRESHOLD =", WORD_COUNT_THRESHOLD)
    word2index, index2word, WORD_LIST, mapping_index_dict = modify_vocab(WORD_LIST, WORD_FREQ, WORD_COUNT_THRESHOLD)
else:
    mapping_index_dict = None
print("Vocab:", len(WORD_LIST) )


def main():

    # weight_decay=0.
    weight_decay = 1e-6
    do_load_pretrained_embeddings = False
    freeze_embeddings = False

    # Load pre-trained?
    do_load_checkpoint = True

    do_decode_val = True
    do_decode_val_beamsearch = False

    do_plot_attention_masks_on_val = False
    decode_first_batch_only = False

    do_decode_test = False
    do_decode_test_beamsearch = False

    score_captions = False

    beam_size = 10
    # beam_size = int(sys.argv[1])
    use_lm_bigram = False
    use_lm_trigram = False

    if use_lm_bigram or use_lm_trigram:
        # lm_weight = float(sys.argv[3])
        lm_weight = 0.5
    else:
        lm_weight = 0.

    if do_decode_val_beamsearch or do_decode_test_beamsearch:
        if use_lm_bigram:
            print("beam_size:", beam_size, "LM order: 2", "lm_w:", lm_weight)
        elif use_lm_trigram:
            print("beam_size:", beam_size, "LM order: 3", "lm_w:", lm_weight)
        else:
            print("beam_size:", beam_size, "no LM")


    model_dir = "checkpoints/4367_red_2_2__128_64_0.98_False_False_0.0005_1e-06/"
    checkpoint_pathname = model_dir + '40_1.6245147620307074_2.6875626488429742_checkpoint.tar'
    save_dir='checkpoints'
    print("save_dir", save_dir)

    input_dim = 64
    vocab_size = len(WORD_LIST) # seq2seq at WORD LEVEL
    if do_load_pretrained_embeddings:
        emb_fpath = '/tmpdir/pellegri/corpus/clotho-dataset/lm/word2vec_dev_128.pth'
    else:
        emb_fpath = None

    use_spec_augment = False
    use_gumbel_noise = False

    encoder_hidden_dim = 128
    embedding_dim = 128
    value_size, key_size, query_size = [64] * 3  # these could be different from embedding_dim

    # teacher_forcing_ratio = 1.
    teacher_forcing_ratio = float(sys.argv[1]) # 0.98 or 1 when scoring predictions
    config_pBLSTM_str = sys.argv[2:]

    pBLSTM_time_reductions = [int(config_pBLSTM_str[i]) for i in range(len(config_pBLSTM_str))]
    print("config pBLSTM", pBLSTM_time_reductions)
    # nb_pBLSTM_layers = len(pBLSTM_time_reductions) # from 1 to 3


    decoder_hidden_size_1 = 128
    decoder_hidden_size_2 = 64
    # [2,2] 0 --> 2887375 params
    # [2,2] 8 --> 2904015 params

    print("use Gumbel noise", use_gumbel_noise)
    print("use teacher forcing", teacher_forcing_ratio)
    print("use SpecAugment", use_spec_augment)

    if do_decode_val or do_decode_test:

        model = Seq2Seq(input_dim=input_dim, vocab_size=vocab_size, encoder_hidden_dim=encoder_hidden_dim,
                        use_spec_augment=use_spec_augment,
                        embedding_dim=embedding_dim,
                        decoder_hidden_size_1=decoder_hidden_size_1,
                        decoder_hidden_size_2=decoder_hidden_size_2, query_size=query_size,
                        value_size=value_size, key_size=key_size, isAttended=True,
                        pBLSTM_time_reductions=pBLSTM_time_reductions,
                        emb_fpath=emb_fpath, freeze_embeddings=freeze_embeddings,
                        teacher_forcing_ratio=teacher_forcing_ratio,  # beam_size=beam_size, lm_weight=lm_weight,
                        word2index=word2index, return_attention_masks=False, device=DEVICE)

    elif do_decode_val_beamsearch or do_decode_test_beamsearch :
        print("Beam decoding w")
        if use_lm_bigram:
            print(" using 2g LM with lm_w=%.3f"%(lm_weight))
        elif use_lm_trigram:
            print(" using 3g LM with lm_w=%.3f"%(lm_weight))
        else: print(" not using LM")
        print(" bs=", beam_size)

        model = BeamSeq2Seq(input_dim=input_dim, vocab_size=vocab_size,  encoder_hidden_dim=encoder_hidden_dim, use_spec_augment=use_spec_augment,
                        embedding_dim=embedding_dim,
                        decoder_hidden_size_1=decoder_hidden_size_1,
                        decoder_hidden_size_2=decoder_hidden_size_2, query_size=query_size,
                        value_size=value_size, key_size=key_size, isAttended=True,
                            pBLSTM_time_reductions=pBLSTM_time_reductions,
                        teacher_forcing_ratio=teacher_forcing_ratio, beam_size=beam_size, use_lm_bigram=use_lm_bigram, use_lm_trigram=use_lm_trigram, lm_weight=lm_weight,
                            word2index=word2index, index2word=index2word, vocab=WORD_LIST, return_attention_masks=False, device=DEVICE)

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

        start_train_epoch = model_checkpoint["iteration"]

    model = model.to(DEVICE)

    criterion = masked_ce_loss
    nepochs = start_train_epoch

    if do_decode_val_beamsearch or do_decode_test_beamsearch:
        val_batch_size = 1
    else:
        val_batch_size = 100

    print("nepochs", nepochs)
    print("batch_size", val_batch_size)

    model_name='seq2seq'
    corpus_name='clotho'
    lr=0
    params_dict = get_params_dict(model_name, corpus_name, input_dim, vocab_size, embedding_dim, value_size,
                                  pBLSTM_time_reductions, teacher_forcing_ratio, use_gumbel_noise, use_spec_augment,
                                  lr, weight_decay, emb_fpath, freeze_embeddings)

    split = 'clotho_dataset_dev'
    input_field_name = 'features'
    # output_field_name = 'caption'
    # output_field_name = 'caption_ind'
    output_field_name = 'words_ind'
    # output_field_name = 'chars_ind'
    fileid_field_name = 'file_name'

    #!!!! change to True
    load_into_memory = True

    nb_t_steps_pad = 'max'
    has_gt_text = True
    shuffle = False
    drop_last = False
    # input_pad_at='start'
    input_pad_at = 'end'
    output_pad_at = 'end'
    num_workers = 0

    if do_decode_val or do_decode_val_beamsearch:
        split = 'clotho_dataset_eva'
        if score_captions:
            has_gt_text = False
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


    if do_decode_test or do_decode_test_beamsearch:
        split = 'clotho_dataset_test'
        has_gt_text = False

        test_loader = get_clotho_loader(data_dir=data_dir_path,
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

    if do_decode_val :
        print("decoding val subset GREEEDY SEARCH...")
        result_fpath = 'results_decode_val_greedy.txt'
        result_fh = open(result_fpath, "at")

        if do_plot_attention_masks_on_val:
            att_masks, first_batch_text, first_batch_preds_char = decode_val(model, val_loader,
                                                                             criterion, index2word, word2index,
                                                                             decode_first_batch_only=decode_first_batch_only,
                                                                             use_gumbel_noise=False, plot_att=True,  device=DEVICE)
            is_already_text = True
            plot_att_masks_to_png_files(att_masks, first_batch_preds_char, is_already_text, index2word, word2index, save_dir, model_dir, params_dict)
        else:
            if not score_captions:
                captions_pred, captions_gt_indices, all_ids_str = decode_val(model, val_loader, criterion, index2word, word2index,
                                      decode_first_batch_only=decode_first_batch_only, use_gumbel_noise=False, plot_att=False,  device=DEVICE)

                captions_gt = index2words(captions_gt_indices, index2word)
            #
                captions_pred_every_five = captions_pred[::5]
                all_ids_str_every_five = all_ids_str[::5]
                # save_gt_captions(data_dir + "/clotho_captions_evaluation.pkl", captions_gt, all_ids_str_every_five)
                # save_gt_captions(data_dir + "/clotho_captions_evaluation_50.pkl", captions_gt, all_ids_str_every_five)

                # gt_file = "/clotho_captions_evaluation.pkl"
                # print("GT CAPTION FILE:", data_dir +  gt_file)
                # captions_gt = load_gt_captions(data_dir + gt_file, all_ids_str_every_five)

                print("captions_gt_indices", len(captions_gt_indices))
                print("captions_pred", len(captions_pred))

                print("captions_gt", len(captions_gt))
                print("captions_pred_every_five", len(captions_pred_every_five))
                print("file ids every_five", len(all_ids_str_every_five))

                out_csv_fpath=model_dir + "/val_predicted_captions_greedy_NEW.csv"
                write_csv_prediction_file(captions_pred_every_five, all_ids_str_every_five, out_csv_fpath)

                metrics = evaluate_metrics_from_lists(captions_pred_every_five, captions_gt)

                average_metrics= metrics[0]
                print("\n")
                for m in average_metrics.keys():
                    print("%s\t%.3f"%(m, average_metrics[m]))
                result_fh.write("%s\t%.3f\n" % ('SPIDEr', average_metrics['SPIDEr']))

                result_fh.write("%s,%.3f,%s,%s\n" % ("_".join(config_pBLSTM_str),
                                                     average_metrics['SPIDEr'],
                                                     checkpoint_pathname,
                                                        emb_fpath
                                                     ))

                result_fh.close()

            else:
                pred_fpath = 'checkpoints/seq2seq/clotho/best_model/4367_red_2_2__128_64_0.98_False_False_0.0005_1e-06//val_predicted_captions_beamsearch_nolm_bs25_alpha_12.csv'

                wav_id_list, captions_dict_pred = read_csv_prediction_file(pred_fpath)
                print(wav_id_list[0], captions_dict_pred[wav_id_list[0]])

                criterion = masked_ce_loss_per_utt
                test_losses, all_ids_str = score_test_captions(model, criterion, val_loader, captions_dict_pred, index2word,
                                                               word2index, use_gumbel_noise=False, device=DEVICE)
                csv_out_fpath = 'checkpoints/seq2seq/clotho/best_model/4367_red_2_2__128_64_0.98_False_False_0.0005_1e-06//val_predicted_captions_beamsearch_nolm_bs25_alpha_12_scores.csv'

                with open(csv_out_fpath, "wt") as fh:
                    for ind_wav, wav_id in enumerate(all_ids_str):
                        fh.write("%s,%f\n" % (wav_id, test_losses[ind_wav]))

    elif do_decode_val_beamsearch:

        print("decoding val subset BEAM SEARCH...")

        result_fpath = 'results_decode_val_beamsearch.txt'
        result_fh = open(result_fpath, "at")
        captions_pred, captions_gt_indices, all_ids_str = bs_decode_val(model, val_loader, index2word,
                                                                        use_gumbel_noise=use_gumbel_noise,
                                                                        device=DEVICE)


        captions_pred_every_five = captions_pred[::5]
        all_ids_str_every_five = all_ids_str[::5]
        # captions_pred_every_five = captions_pred
        # all_ids_str_every_five = all_ids_str

        # gt_file = "/clotho_captions_evaluation.pkl"
        # print("GT CAPTION FILE:", data_dir + gt_file)
        # captions_gt = load_gt_captions(data_dir + gt_file, all_ids_str_every_five)

        captions_gt = index2words(captions_gt_indices, index2word)

        print("captions_gt_indices", len(captions_gt_indices))
        print("captions_gt", len(captions_gt))
        print("captions_pred", len(captions_pred))

        print("captions_pred_every_five", len(captions_pred_every_five))
        print("file ids every_five", len(all_ids_str_every_five))

        print("\n")
        if use_lm_bigram:
            out_csv_fpath = model_dir + "/val_predicted_captions_beamsearch_lm_%.2f_2g.csv"%lm_weight
        elif use_lm_trigram:
            out_csv_fpath = model_dir + "/val_predicted_captions_beamsearch_lm_%.2f_3g.csv" % lm_weight
        else:
            out_csv_fpath = model_dir + "/val_predicted_captions_beamsearch_nolm_bs%d_alpha_12.csv"%(beam_size)

        write_csv_prediction_file(captions_pred_every_five, all_ids_str_every_five, out_csv_fpath)

        if not decode_first_batch_only:

            metrics = evaluate_metrics_from_lists(captions_pred_every_five, captions_gt)
            print("\n")
            average_metrics = metrics[0]
            for m in average_metrics.keys():
                # print("%s\t%.3f" % (m, average_metrics[m]))
                print("%.3f" % (average_metrics[m]))

            result_fh.write("%s,%d,%.3f,%s,%.2f\n" % ("_".join(config_pBLSTM_str),
                                                 n_attn_heads,
                                                 average_metrics['SPIDEr'],
                                                 checkpoint_pathname,
                                                 lm_weight
                                                 ))
            result_fh.close()

    elif do_decode_test:
        if not score_captions:
            print("decoding test subset (greedy)...")
            captions_pred, all_ids_str = decode_test(model, test_loader, index2word, use_gumbel_noise=False, device=DEVICE)

            print("captions_pred", len(captions_pred))
            out_csv_fpath = model_dir + "/test_predicted_captions_greedy.csv"
            write_csv_prediction_file(captions_pred, all_ids_str, out_csv_fpath)
        else:
            pred_fpath = '../dcase2020_challenge_submission_task6_thomas_pellegrini/Pellegrini_IRIT_task6_3/test_predicted_captions_beamsearch_lm_0.50_2g.csv'

            wav_id_list, captions_dict_pred = read_csv_prediction_file(pred_fpath)
            print(wav_id_list[0], captions_dict_pred[wav_id_list[0]])

            criterion = masked_ce_loss_per_utt
            test_losses, all_ids_str = score_test_captions(model, criterion, test_loader, captions_dict_pred, index2word, word2index, use_gumbel_noise=False, device=DEVICE)
            csv_out_fpath = '../dcase2020_challenge_submission_task6_thomas_pellegrini/Pellegrini_IRIT_task6_3/scores_per_utt_sub3.csv'

            with open(csv_out_fpath, "wt") as fh:
                for ind_wav, wav_id in enumerate(all_ids_str):
                    fh.write("%s,%f\n"%(wav_id, test_losses[ind_wav]))

    elif do_decode_test_beamsearch:

        captions_pred, all_ids_str = bs_decode_test(model, test_loader, index2word, use_gumbel_noise=False, device=DEVICE)
        print("test captions_pred", len(captions_pred))

        print("\n")
        if use_lm_bigram:
            out_csv_fpath = model_dir + "/test_predicted_captions_beamsearch_lm_%.2f_2g.csv" % lm_weight
        elif use_lm_trigram:
            out_csv_fpath = model_dir + "/test_predicted_captions_beamsearch_lm_%.2f_3g.csv" % lm_weight
        else:
            out_csv_fpath = model_dir + "/test_predicted_captions_beamsearch_nolm_bs25_alpha12.csv"

        write_csv_prediction_file(captions_pred, all_ids_str, out_csv_fpath)


if __name__ == '__main__':
    main()

