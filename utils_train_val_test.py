"""Helper functions to train a model on one epoch, decode the val subset, greedy decode, beam search decode and score predicted captions"""

import time
import torch

from utils import captions2index

__author__ = "Thomas Pellegrini - 2020"

def train(model, train_loader, criterion, optimizer, epoch, pretrain_decoder=False, use_gumbel_noise=False, device='cpu'):

    start = time.time()

    model.train()

    train_loss_avg = 0

    for batch_idx, train_batch in enumerate(train_loader):

        # print("batch nb", batch_idx)

        # Use torch.autograd.set_detect_anomaly(True) to get notices about gradient explosion
        torch.autograd.set_detect_anomaly(True)

        # Set the inputs to the device
        speech_batch, text_batch, speech_lengths, text_lengths, ids_batch = train_batch
        speech_batch = speech_batch.to(device)
        text_batch = text_batch.to(device)
        speech_lengths = speech_lengths.to(device)
        text_lengths = text_lengths.to(device)

        # Initialising gradients to zero
        optimizer.zero_grad()

        # Pass the inputs, and length of speech into the model
        probs = model(speech_batch, speech_lengths, text_input=text_batch, pretrain_decoder=pretrain_decoder, use_gumbel_noise=use_gumbel_noise, isTrain=True)
        torch.cuda.empty_cache()

        # text_batch: B,padded_length
        # we want to predict the next word so text_batch[:,1:] and text_lengths-1
        loss = criterion(probs, text_batch[:,1:].contiguous(), text_lengths-1, device)
        
        # Run the backward pass on the masked loss
        loss.backward()

        # Clip the gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)

        # Update weights and optimizer params
        optimizer.step()

        # Add masked loss
        train_loss_avg += loss.item()

        # Print the training loss after every N batches
        if batch_idx % 50 == 0: print("   batch %d\tloss: %.5f"%(batch_idx, loss))

    end = time.time()
    train_loss_avg /= batch_idx
    print("  train epoch %d:\ttrain_loss: %.5f\ttook %.1f sec"%(epoch, train_loss_avg, end-start))
    return train_loss_avg


def val(model, val_loader, criterion, epoch, pretrain_decoder=False, use_gumbel_noise=False, print_captions=False, index2word=None, device='cpu'):
    model.eval()
    start = time.time()

    val_loss_avg = 0
    batch_idx = 0

    for batch_idx, val_batch in enumerate(val_loader):
        with torch.no_grad():

            speech_batch, text_batch, speech_lengths, text_lengths, ids_batch = val_batch
            speech_batch = speech_batch.to(device)
            text_batch = text_batch.to(device)
            speech_lengths = speech_lengths.to(device)
            text_lengths = text_lengths.to(device)

            if batch_idx == 0:
                first_batch_text = text_batch.clone().detach().cpu()

            if print_captions:
                probs = model(speech_batch, speech_lengths, text_input=None, pretrain_decoder=pretrain_decoder,
                              use_gumbel_noise=use_gumbel_noise, isTrain=False)  # size: B, T, Vocab

                preds_words = greedy_captioning(probs, index2word)
                print(" len(predictions)", len(preds_words))
                for i in range(20):
                    print_predicted_and_gt_utterance(preds_words, first_batch_text, i, index2word)
                print_captions=False

            probs = model(speech_batch, speech_lengths, text_input=text_batch, pretrain_decoder=pretrain_decoder, use_gumbel_noise=use_gumbel_noise, isTrain=True)

            torch.cuda.empty_cache()

            loss = criterion(probs, text_batch[:,1:].contiguous(), text_lengths-1, device)

            val_loss_avg += loss.item()

            batch_idx += 1
            
    end = time.time()
    val_loss_avg /= batch_idx
    print("  val epoch %d:\tval_loss: %.5f\ttook %.1f sec"%(epoch, val_loss_avg, end-start))

    return val_loss_avg


def print_predicted_and_gt_utterance(prediction_char, gt_text_batch, index, index2word, is_beamsearch=True):
    print(" hyp :",  prediction_char[index])
    gt = ''
    is_not_eos = True
    # print(index, gt_text_batch)
    i = 0
    if is_beamsearch:
        while is_not_eos and i < gt_text_batch[index].shape[0]:
            c = gt_text_batch[index][i]
            c = index2word[c.item()]
            gt += c
            gt += ' '
            is_not_eos = c != '<eos>'
            i += 1
    else:
        while is_not_eos and i < gt_text_batch[index].shape[0]:
            c = gt_text_batch[index, i]
            c = index2word[c.item()]
            gt += c
            gt += ' '
            is_not_eos = c != '<eos>'
            i += 1

    print(" gt:", gt, "\n")


def greedy_captioning(probs_tensor, index2word):
    """greedy decoding on a probability pytorch tensor"""

    preds = torch.argmax(probs_tensor, dim=-1).detach().cpu().numpy()
    preds_word = []

    # print(index2word)

    for i in range(preds.shape[0]):
        t = 0
        is_not_eos = True
        pred_utt = '<sos> '
        while is_not_eos and t < preds.shape[1]:
            c = index2word[preds[i, t]]
            # if c!='<sos>' : pred_utt += c
            pred_utt += c
            pred_utt += ' '
            is_not_eos = c != '<eos>'
            t += 1
        preds_word.append(pred_utt.replace('<sos> ', '').replace(' <eos> ', ''))
    return preds_word


def decode_val(model, data_loader, criterion, index2word, word2index, decode_first_batch_only = False, plot_att=False, use_gumbel_noise=False, device='cpu'):

    model.eval()
    start = time.time()

    if decode_first_batch_only:
        val_batch = next(iter(data_loader))
        val_att_masks = []

        with torch.no_grad():

            speech_batch, text_batch, speech_lengths, text_lengths, ids_batch = val_batch
            speech_batch = speech_batch.to(device)
            text_batch = text_batch.to(device)
            first_batch_text = text_batch.clone().detach().cpu()
            speech_lengths = speech_lengths.to(device)
            text_lengths = text_lengths.to(device)

            if plot_att:
                print("Val: getting att_masks")
                print("size speech_batch", speech_batch.size())
                print("size text_batch", text_batch.size())
                probs, att_masks = model(speech_batch, speech_lengths, text_input=None,
                                         use_gumbel_noise=use_gumbel_noise, isTrain=False, return_attention_masks=True)
                val_att_masks.append(att_masks)

            else:
                probs = model(speech_batch, speech_lengths, text_input=None,
                              use_gumbel_noise=use_gumbel_noise, isTrain=False)  # size: B, T, Vocab


        preds_words = greedy_captioning(probs, index2word)
        print(" len(predictions)", len(preds_words))
        # for i in range(50):
        #     print_predicted_and_gt_utterance(preds_words, first_batch_text, i, index2word)

        if len(val_att_masks)>0:
            return val_att_masks, first_batch_text, preds_words

    else:
        # decode the whole validation subset
        all_preds_words = []
        gt_words_indices = []
        all_ids_str = []

        for batch_idx, val_batch in enumerate(data_loader):

            with torch.no_grad():
                speech_batch, text_batch, speech_lengths, text_lengths, ids_batch = val_batch

                if batch_idx == 0:
                    first_batch_text = text_batch

                gt_words_indices.extend(text_batch.clone().detach().cpu())

                # append file ids for scoring later (outside this function)
                all_ids_str.extend(ids_batch)

                speech_batch = speech_batch.to(device)
                text_batch = text_batch.to(device)
                speech_lengths = speech_lengths.to(device)
                # text_lengths = text_lengths.to(device)

                probs = model(speech_batch, speech_lengths, text_input=None,
                                  use_gumbel_noise=use_gumbel_noise, isTrain=False) # size: B, T, Vocab

                # ind1 = torch.argmax(probs[0][0]).item()
                # ind2 = torch.argmax(probs[0][1]).item()

                # greedy search
                preds_words = greedy_captioning(probs, index2word)
                all_preds_words.extend(preds_words)

                torch.cuda.empty_cache()

        print(" len(all_preds_words)", len(all_preds_words))

        # for i in range(0,50,5):
        #     print_predicted_and_gt_utterance(all_preds_words, first_batch_text, i, index2word)


        end = time.time()
        print(" took %.1f sec"%(end-start))
        return all_preds_words, gt_words_indices, all_ids_str


def decode_test(model, data_loader, index2word, use_gumbel_noise=False, device='cpu'):

    model.eval()
    start = time.time()

    # decode the whole test subset
    all_preds_words = []
    all_ids_str = []

    for batch_idx, test_batch in enumerate(data_loader):

        with torch.no_grad():
            speech_batch, speech_lengths, ids_batch = test_batch

            # append file ids for scoring later (outside this function)
            all_ids_str.extend(ids_batch)

            speech_batch = speech_batch.to(device)
            speech_lengths = speech_lengths.to(device)

            probs = model(speech_batch, speech_lengths, text_input=None,
                              use_gumbel_noise=use_gumbel_noise, isTrain=False) # size: B, T, Vocab

            # greedy search
            preds_words = greedy_captioning(probs, index2word)
            all_preds_words.extend(preds_words)

            torch.cuda.empty_cache()
            # if batch_idx == 0: break
            # if batch_idx % 2 == 0: print("   batch %d\tloss: %.3f"%(batch_idx, loss))

    end = time.time()
    print(" took %.1f sec"%(end-start))
    return all_preds_words, all_ids_str


def score_test_captions(model, criterion, data_loader, captions_dict_pred, index2word, word2index, use_gumbel_noise=False, device='cpu'):

    model.eval()
    start = time.time()

    # decode the whole test subset
    all_ids_str = []
    test_losses = []

    for batch_idx, test_batch in enumerate(data_loader):

        with torch.no_grad():
            speech_batch, speech_lengths, ids_batch = test_batch

            pseudo_gt_captions = [captions_dict_pred[fid] for fid in ids_batch]
            text_batch, text_lengths = captions2index(pseudo_gt_captions, word2index)
            # print(text_batch)
            # print("text_lengths", text_lengths)



            speech_batch = speech_batch.to(device)
            speech_lengths = speech_lengths.to(device)
            text_batch = text_batch.to(device)
            text_lengths = text_lengths.to(device)

            probs = model(speech_batch, speech_lengths, text_input=text_batch, pretrain_decoder=False,
                          use_gumbel_noise=use_gumbel_noise, isTrain=True)

            losses = criterion(probs, text_batch[:,1:].contiguous(), text_lengths-1, device)

            test_losses.extend(losses.tolist())
            # append file ids for scoring later (outside this function)
            all_ids_str.extend(ids_batch)

            torch.cuda.empty_cache()
            # if batch_idx == 4: break
            if batch_idx % 1 == 0: print("   batch %d"%batch_idx)

    end = time.time()
    print(" took %.1f sec"%(end-start))
    return test_losses, all_ids_str



def bs_decode_val(model, data_loader, index2word, use_gumbel_noise=False, device='cpu'):

    model.eval()
    start = time.time()

    all_preds_words = []
    gt_words_indices = []
    all_ids_str = []
    # decode the whole validation subset

    for batch_idx, val_batch in enumerate(data_loader):

        with torch.no_grad():
            speech_batch, text_batch, speech_lengths, text_lengths, ids_batch = val_batch

            gt_words_indices.extend(text_batch.clone().detach().cpu())

            # append file ids for scoring later (outside this function)
            all_ids_str.extend(ids_batch)

            speech_batch = speech_batch.to(device)
            speech_lengths = speech_lengths.to(device)

            hyps = model(speech_batch, speech_lengths, text_input=None,
                         use_gumbel_noise=use_gumbel_noise, isTrain=False)  # size: B, T, Vocab

            hyp_words = [" ".join([index2word[ind] for ind in hyp]).replace(" <eos>", "") for hyp in hyps]
            del hyps

            all_preds_words.append(hyp_words[0])

            if batch_idx % 100 == 0: print("BS for batch i:", batch_idx)

            torch.cuda.empty_cache()

    print(" len(all_preds_words)", len(all_preds_words))
    print(" len(gt_words_indices)", len(gt_words_indices))
    # print("first_batch_text", first_batch_text)

    # for i in range(0,50,5):
    #     # print(i)
    #     # print(gt_words_indices[i])
    #     print_predicted_and_gt_utterance(all_preds_words, gt_words_indices, i, index2word, is_beamsearch=True)


    end = time.time()
    print(" took %.1f sec"%(end-start))

    return all_preds_words, gt_words_indices, all_ids_str


def bs_decode_test(model, data_loader, index2word, use_gumbel_noise=False, device='cpu'):

    model.eval()
    start = time.time()

    all_preds_words = []
    all_ids_str = []
    # decode the whole test subset w beamsearch

    for batch_idx, val_batch in enumerate(data_loader):

        with torch.no_grad():
            speech_batch, speech_lengths, ids_batch = val_batch

            # append file ids for scoring later (outside this function)
            all_ids_str.extend(ids_batch)

            speech_batch = speech_batch.to(device)
            speech_lengths = speech_lengths.to(device)

            hyps = model(speech_batch, speech_lengths, text_input=None,
                         use_gumbel_noise=use_gumbel_noise, isTrain=False)  # size: B, T, Vocab
            # print("hyps", hyps)

            hyp_words = [" ".join([index2word[ind] for ind in hyp]).replace(" <eos>", "") for hyp in hyps]
            del hyps

            all_preds_words.append(hyp_words[0])

            if batch_idx % 100 == 0: print("BS for batch i:", batch_idx)

            torch.cuda.empty_cache()

    print(" len(all_preds_words)", len(all_preds_words))

    end = time.time()
    print(" took %.1f sec"%(end-start))

    return all_preds_words, all_ids_str
