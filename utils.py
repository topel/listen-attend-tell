import torch
import os
import matplotlib.pyplot as plt

from torch import cat as pt_cat, LongTensor

import pickle

__author__ = "Thomas Pellegrini - 2020"

def get_params_dict(
        model_name,
        corpus_name,
        input_dim,
        vocab_size,
        hidden_dim,
        value_size,
        pBLSTM_time_reductions,
        teacher_forcing_ratio,
        use_gumbel_noise,
        use_spec_augment,
        lr,
        weight_decay,
        emb_fpath,
        freeze_embeddings
):
    if emb_fpath is not None:
        emb_basename=os.path.basename(emb_fpath)
        emb_basename = os.path.splitext(emb_basename)[0]

        if freeze_embeddings: freeze_str = 'frozen'
        else: freeze_str = 'learnable'

    else:
        emb_basename =''
        freeze_str = ''


    params_dict = {
        "model_name": model_name,
        "corpus_name":corpus_name,
        "input_dim": input_dim,
        "vocab_size":vocab_size,
        "hidden_dim":hidden_dim,
        "value_size":value_size,
        "pBLSTM_time_reductions":pBLSTM_time_reductions,
        "teacher_forcing_ratio":teacher_forcing_ratio,
        "use_gumbel_noise":use_gumbel_noise,
        "use_spec_augment":use_spec_augment,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "emb_basename": emb_basename,
        "freeze_str":freeze_str
    }

    return params_dict


def save_checkpoint(
            save_dir,
            model,
            model_optimizer,
            iteration,
            train_loss,
            val_loss,
            params_dict,
            do_pretrain_decoder_as_an_lm=False
    ):

    time_reduction_string = 'red_'
    for el in params_dict['pBLSTM_time_reductions']:
        time_reduction_string += str(el) + '_'
    if do_pretrain_decoder_as_an_lm:
        model_dir = os.path.join(save_dir,
                                 'decoder_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(params_dict['vocab_size'],
                                                            time_reduction_string,
                                                         params_dict['hidden_dim'],
                                                         params_dict['value_size'],
                                                         params_dict['teacher_forcing_ratio'],
                                                         params_dict['use_gumbel_noise'],
                                                         params_dict['use_spec_augment'],
                                                         params_dict['learning_rate'],
                                                         params_dict['weight_decay'],
                                                         params_dict['emb_basename'],
                                                         params_dict['freeze_str']))
    else:
        model_dir = os.path.join(save_dir,
                                 '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_dev_eva_conv'.format(params_dict['vocab_size'],
                                                                     time_reduction_string,
                                                                     params_dict['hidden_dim'],
                                                                     params_dict['value_size'],
                                                                     params_dict['teacher_forcing_ratio'],
                                                                     params_dict['use_gumbel_noise'],
                                                                     params_dict['use_spec_augment'],
                                                                     params_dict['learning_rate'],
                                                                     params_dict['weight_decay'],
                                                                    params_dict['emb_basename'],
                                                                     params_dict['freeze_str']))

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if do_pretrain_decoder_as_an_lm:
        print("  saving decoder checkpoint")
        torch.save(
            {
                "iteration": iteration,
                "model": model.decoder.state_dict(),
                "model_optim": model_optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss
            },
            os.path.join(model_dir, "{}_{:.3f}_{:.3f}_checkpoint.tar".format(iteration, train_loss, val_loss))
        )
    else:
        print("  saving checkpoint")
        torch.save(
            {
                "iteration": iteration,
                "model": model.state_dict(),
                "model_optim": model_optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss
            },
            os.path.join(model_dir, "{}_{:.3f}_{:.3f}_checkpoint.tar".format(iteration, train_loss, val_loss))
        )

    print("checkpoint saved into:", model_dir)
    return model_dir


def plot_att_masks_to_png_files(att_masks, text_batch, is_already_text, index2letter, letter2index, save_dir, model_dir, params_dict):
    nb_plots = 10

    #  size speech_batch torch.Size([576, 32, 40])
    #  size text_batch torch.Size([32, 94])
    #  SEQ2SEQ, size of att torch.Size([32, 93, 72])
    # image : 93, 72 : text, pBLSTM output time duration

    print("nb of batches:", len(att_masks))
    print("nb of masks per batch (except last batch):", len(att_masks[0]))

    assert os.path.exists(model_dir), "ERROR: model_dir does not exist: cannot plot att masks"
    print("size of the ten first masks:") # (154, 130): padded text, acoustic frames but after pBLSTM reduction

    for i in range(nb_plots):
        mask = att_masks[0][i].clone().detach().cpu().numpy()
        hauteur, largeur = mask.shape
        print(i,  mask.shape)
        print(type(text_batch))
        if is_already_text:
            text = text_batch[i].split(" ")
        else:
            text = text_batch[i].tolist()
            # text = [index2letter[el] for el in text if el != letter2index['<pad>'] and el != letter2index['<sos>']and el != letter2index['<eos>']]
            text = [index2letter[el] for el in text if el != letter2index['<sos>']and el != letter2index['<eos>']]
        print(text)
        # text = text[::-1]
        plt.figure(figsize=(14,10))
        # plt.imshow(mask[:20, :70])
        # plt.yticks(range(20), text[:20]) #, rotation=90)
        plt.imshow(mask, aspect='auto')
        plt.yticks(range(len(text)), text)  # , rotation=90)
        plt.xlabel("Time (pBLSTM encoder output)")
        plt.savefig(os.path.join(model_dir, '{}.png'.format(i)))

    print("10 plots saved into", model_dir)


def index2words(text_tensor, index2unit):
    """index2words from a list of pytorch tensor of indices"""

    text_in_units = []

    # print(index2letter)
    five_counter = 0
    five_captions = []
    # print("len(text_tensor)", len(text_tensor))
    for i in range(len(text_tensor)):
        t = 0
        is_not_eos = True
        indice_utt = text_tensor[i]
        # print("indice_utt", indice_utt)
        pred_utt = ''
        while is_not_eos and t < indice_utt.size(0):
            c = index2unit[indice_utt[t].item()]
            # if c!='<sos>' : pred_utt += c
            pred_utt += c
            pred_utt += ' '
            is_not_eos = c != '<eos>'
            t += 1
        five_captions.append(pred_utt.replace('<sos> ', '').replace(' <eos> ', ''))
        # text_in_units.append(five_captions)

        five_counter += 1
        if five_counter == 5:
            text_in_units.append(five_captions)
            five_captions = []
            five_counter=0

    return text_in_units


def captions2index(text_list, word2index):
    """returns captions in integers and text lengths"""
    index_list, text_length = [], []
    # print("len(text_list[0]", len(text_list[0].split(" ")))

    # minibatches are ordered by decreasing AUDIO feature length,
    # not in decreasing nb of words in the corresponding captions
    # hence, the max length in words may be any of the captions
    # of a minibatch:
    max_length = max([len(t.split(" ")) for t in text_list])
    # print("max_length", max_length)

    for caption in text_list:
        seq = [word2index[w] for w in caption.split(" ")]
        text_length.append(len(seq))
        while len(seq)<max_length: seq.extend([word2index["<eos>"]])
        index_list.append(LongTensor(seq).unsqueeze_(0))

    # index_list = LongTensor(index_list)
    # print(index_list)

    return pt_cat(index_list), LongTensor(text_length)


def write_csv_prediction_file(captions_list_pred, wav_id_list, out_csv_fpath):
    with open(out_csv_fpath, "wt") as fh:
        fh.write("file_name\tcaption_predicted\n")
        print(" write_csv_prediction_file")
        print("  ", wav_id_list[0], captions_list_pred[0])
        for id, caption in zip(wav_id_list,captions_list_pred):
            fh.write("%s\t%s\n"%(id,caption))

    print("INFO: predicted captions saved to file:", out_csv_fpath)


def read_csv_prediction_file(in_csv_fpath, add_sos_eos=True):
    wav_id_list = []
    captions_dict_pred = {}
    firstLine=True
    with open(in_csv_fpath, "rt") as fh:
        for ligne in fh:
            if firstLine:
                firstLine=False
                continue
            tab = ligne.rstrip().split("\t")
            wav_id_list.append(tab[0])
            if add_sos_eos:
                captions_dict_pred[tab[0]] = "<sos> " + tab[1] + " <eos>"
                # captions_list_pred.append("<sos> " + tab[1] + " <eos>" )
            else:
                captions_dict_pred[tab[0]] = tab[1]
                # captions_list_pred.append(tab[1])
        # print("  ", wav_id_list[0], captions_list_pred[0])

    print("INFO: predicted captions read from file:", in_csv_fpath)
    return wav_id_list, captions_dict_pred


def save_gt_captions(pickle_file_path, captions_gt, all_ids_str):
    dico = {}
    for i,fid in enumerate(all_ids_str):
        dico[fid] = captions_gt[i]
    pickle.dump( dico, open(pickle_file_path, "wb"))
    print("written captions to file:", pickle_file_path)


def load_gt_captions(caption_pickle_fpath, all_ids_str):
    dico = pickle.load(open(caption_pickle_fpath, "rb"))
    captions = []
    for i,id in enumerate(all_ids_str):
        captions.append(dico[id])

    return captions

