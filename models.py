"""Provides the model components: Seq2Seq and BeamSeq2Seq with Attention, pBLSTM, Encoder, Decoder,
                                    Seq2Seq, BeamDecoder, and the loss function masked_ce_loss"""

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F_torch
import random
from pynlpl.lm import lm as lm_object
import numpy as np
from augmentation import SpecAugmentation

import socket
host_name = socket.gethostname()
print(host_name)

__author__ = "Thomas Pellegrini - 2020"

lm_dir='../clotho-dataset/lm/'


def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Attention(nn.Module):
    '''
    Attention is calculated using key, value and query from Encoder and decoder.
    Below are the set of operations you need to perform for computing attention:
        energy = bmm(key, query)
        attention = softmax(energy)
        context = bmm(attention, value)
    returns:
      - a context vector of size B,H
      - a masked attention vector of size B,T
    e.g., torch.Size([3, 128]) torch.Size([3, 8])
    '''

    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, query, key, value, mask):
        '''
        :param query :(B, context_size) Query is the output of LSTMCell from Decoder
        :param key: (T, B, key_size) Key Projection from Encoder all time steps
        :param value: (T, B, value_size) Value Projection from Encoder all time steps
        :param mask: (B, 1, T//pBLSTMfactor) boolean mask
        :return output: Attended Context, 1,B,H
        :return attention_mask: Attention mask that can be plotted, B,T
        '''
        query = query.unsqueeze(1)  # necessary to use bmm: 3-d tensors are expected by bmm

        key = key.transpose(0, 1).contiguous()  # T,B,h --> B,T,h, necessary to use bmm, needs batch size first
        value = value.transpose(0, 1).contiguous()  # T,B,h --> B,T,h, necessary to use bmm, needs batch size first
        
        energy = torch.bmm(query, key.transpose(1, 2).contiguous())  # energy size: B, 1, T
        attention = F_torch.softmax(energy, dim=2)  # softmax on the time axis T

        masked_attention = mask * attention
        masked_attention /= torch.sum(masked_attention, dim=2).unsqueeze_(1) # normalize(masked_attention)to get one values if sum columns
        
        # masked_attention.unsqueeze_(dim=1) # ---> B,1,T
        context = torch.bmm(masked_attention, value)  # B,1,T and B,T,h --> B,1,h
        context = context.transpose(0, 1).contiguous()  # 1,B,h

        context = torch.squeeze(context)
        masked_attention = torch.squeeze(masked_attention)
        return context, masked_attention
        # return context.squeeze_(), masked_attention.squeeze_()


class pBLSTM(nn.Module):
    '''
    Pyramidal BiLSTM
    '''

    def __init__(self, input_dim, hidden_dim, reduction_time_factor=2, boolean_use_pack_padded_sequences=True):
        super(pBLSTM, self).__init__()

        self.blstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True)
        self.reduction_time_factor = reduction_time_factor

        assert self.reduction_time_factor == 1 or self.reduction_time_factor == 2 or self.reduction_time_factor == 4 or self.reduction_time_factor == 8, "reduction_time_factor can be 2, 4 or 8."
        print("reduction_time_factor", self.reduction_time_factor)
        self.boolean_use_pack_padded_sequences = boolean_use_pack_padded_sequences

        self.dropout_layer = nn.Dropout(p=0.1)

    def forward(self, x, use_dropout=False):
        '''
        :param x :(T, B, F) input to the pBLSTM, batch of pack_padded sequences, T must be even
        :return output: (T//reduc, B, H*reduc) encoded sequence from pyramidal Bi-LSTM
        '''

        if self.boolean_use_pack_padded_sequences:
            x, lens = rnn_utils.pad_packed_sequence(x)
        else:
            lens = torch.tensor([x.size(0)] * x.size(1), dtype=torch.int)

        while x[:, 0, :].size(0) % self.reduction_time_factor != 0:
            # removing last element from tensor to get even time length
            x = x[:-1]
            lens -= 1

        assert lens[-1] >= self.reduction_time_factor, "smallest element in tensor is too small for this reduction rate"

        x = torch.transpose(x, 0, 1).contiguous()
        B, T, F = x.size()
        x = x.view(B, T // self.reduction_time_factor, F * self.reduction_time_factor)
        x = torch.transpose(x, 0, 1).contiguous()

        if use_dropout:
            x = self.dropout_layer(x)

        x = rnn_utils.pack_padded_sequence(x, lengths=lens // self.reduction_time_factor, batch_first=False,
                                           enforce_sorted=True)
        output, _ = self.blstm(x)

        return output, lens // self.reduction_time_factor


class Encoder(nn.Module):
    '''
    Encoder takes the recordings as inputs and returns key and value.
    Key and value are projections of the output from pBLSTM network.
    '''

    def __init__(self, input_dim, hidden_dim, value_size=128, key_size=128, pBLSTM_time_reductions=[2], use_spec_augment=False, use_conv_blocks_in_encoder=False):
        super(Encoder, self).__init__()

        self.use_dropout = True

        self.use_spec_augment = use_spec_augment
        print("Encoder, using spec augment:", use_spec_augment)

        self.dropout_layer = nn.Dropout(p=0.1)
        print("Encoder, using dropout:", self.use_dropout, str(0.1))

        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
            freq_drop_width=4, freq_stripes_num=2)

        self.use_conv_blocks_in_encoder = use_conv_blocks_in_encoder

        if self.use_conv_blocks_in_encoder:

            self.conv_block = nn.Sequential(
                nn.Conv2d(1, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                # nn.Conv2d(64, 1, 3, padding=1),
                # nn.ReLU(),
                nn.MaxPool2d(2, 2)
            )


        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True)

        nb_pBLSTM_layers = len(pBLSTM_time_reductions)
        self.nb_pBLSTM_layers = nb_pBLSTM_layers

        reduction_time_factor = pBLSTM_time_reductions[0]
        self.pblstm1 = pBLSTM(input_dim=hidden_dim * 2 * reduction_time_factor, hidden_dim=hidden_dim,
                              reduction_time_factor=reduction_time_factor)

        if nb_pBLSTM_layers == 1:
            print("Encoder has one pBLSTM layers")
            print(" hidden_dim", hidden_dim)
        if nb_pBLSTM_layers == 2:
            print("Encoder has two pBLSTM layers")
            print(" hidden_dim", hidden_dim)
            reduction_time_factor = pBLSTM_time_reductions[1]
            self.pblstm2 = pBLSTM(input_dim=hidden_dim * 2 * reduction_time_factor, hidden_dim=hidden_dim,
                                  reduction_time_factor=reduction_time_factor)
            print(" hidden_dim", hidden_dim)

        elif nb_pBLSTM_layers == 3:
            print("Encoder has three pBLSTM layers")
            print(" hidden_dim", hidden_dim)
            reduction_time_factor = pBLSTM_time_reductions[1]
            self.pblstm2 = pBLSTM(input_dim=hidden_dim * 2 * reduction_time_factor, hidden_dim=hidden_dim,
                                  reduction_time_factor=reduction_time_factor)
            print(" hidden_dim", hidden_dim)

            reduction_time_factor = pBLSTM_time_reductions[2]
            self.pblstm3 = pBLSTM(input_dim=hidden_dim * 2 * reduction_time_factor, hidden_dim=hidden_dim,
                                  reduction_time_factor=reduction_time_factor)

        self.key_network = nn.Linear(hidden_dim * 2, value_size)
        self.value_network = nn.Linear(hidden_dim * 2, key_size)

    def forward(self, x, lens):
        """x: padded tensor with sequences ordered by decreasing length, size: TxBxF
          lens: list of the corresponding utterance lenghts in x
        """
        # T, B, F = x.size()
        # T_init, B, F_init = x.size()
        # print("Encoder x on GPU?:", x.device.index, "lens on GPU?", lens.device.index)

        # careful: spec_augmenter modifies x in-place
        if self.use_spec_augment:
            print("USING SPEC AUG")
            x = self.spec_augmenter(x.transpose(0,1).unsqueeze(1))
            x = x.squeeze(1).transpose(0,1)

        if self.use_dropout:
            x = self.dropout_layer(x)


        # if self.use_conv_blocks_in_encoder:
        #     x = x.transpose(0, 1).contiguous()
        #
        #     x = torch.unsqueeze(x, dim=1)
        #     # x = self.conv1(x)
        #     # x = self.bn1(x)
        #     # x = F_torch.relu(x)
        #     # x = self.pool(x)
        #     # x = self.conv2(x)
        #     # x = self.bn2(x)
        #     # x = F_torch.relu(x)
        #     # x = self.pool(x)
        #     x = self.conv_block(x)
        #     B, C, T, F = x.size()
        #     x = x.view(B, x.size(2), x.size(1) * x.size(3)).contiguous()
        #
        #     x1 = F_torch.adaptive_max_pool2d(x, (T, F_init // 2))
        #     x2 = F_torch.adaptive_avg_pool2d(x, (T, F_init // 2))
        #     x = torch.cat((x1, x2), dim=-1)
        #     x = x.transpose(0, 1).contiguous()
        #
        #     lens //= 4

        T, B, F = x.size()

        rnn_inp = rnn_utils.pack_padded_sequence(x, lengths=lens, batch_first=False, enforce_sorted=True)
        outputs, _ = self.lstm(rnn_inp)
        # tmp, _ = rnn_utils.pad_packed_sequence(outputs)
        # print("size after non-pyramidal blstm", tmp.size())

        ### Use the outputs and pass it through the pBLSTM blocks! ###
        outputs, new_lengths = self.pblstm1(outputs, self.use_dropout)
        # print("outputs", outputs.size())

        if self.nb_pBLSTM_layers > 1:
            outputs, new_lengths = self.pblstm2(outputs, self.use_dropout)
            # tmp, _ = rnn_utils.pad_packed_sequence(outputs)
            # print("after pblstm2", tmp.size())

        if self.nb_pBLSTM_layers > 2:
            outputs, new_lengths = self.pblstm3(outputs, self.use_dropout)
            # tmp, _ = rnn_utils.pad_packed_sequence(outputs)
            # print("after pblstm3", tmp.size())

        linear_input, _ = rnn_utils.pad_packed_sequence(outputs)
        linear_input = linear_input.contiguous()
        linear_input = linear_input.view(-1, linear_input.shape[2])
        keys = self.key_network(linear_input)
        values = self.value_network(linear_input)

        # reshape to get 3-d tensors with T x B x h
        keys = keys.view(-1, B, keys.size(1))
        values = values.view(-1, B, values.size(1))

        return keys, values, new_lengths


class Decoder(nn.Module):
    """
    Greedy decoder
    """
    def __init__(self, vocab_size, embedding_dim=128, decoder_hidden_size_1=128, decoder_hidden_size_2=128,
                 query_size=128, value_size=128, key_size=128, emb_fpath=None, freeze_embeddings=False, isAttended=False,
                 teacher_forcing_ratio=0.9, word2index=None, device='cpu'):
        super(Decoder, self).__init__()

        if emb_fpath is None:
            print(" using random learnable emb")
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=word2index['<eos>'])
        else:
            if freeze_embeddings:
                print(" using pretrained frozen emb from", emb_fpath)
            else:
                print(" using pretrained LEARNABLE emb from", emb_fpath)
            pretrained_emb = torch.load(emb_fpath)
            self.embedding = nn.Embedding.from_pretrained(pretrained_emb, freeze=freeze_embeddings)

        self.lstm1 = nn.LSTMCell(input_size=embedding_dim + value_size, hidden_size=decoder_hidden_size_1)
        self.lstm2 = nn.LSTMCell(input_size=decoder_hidden_size_1, hidden_size=decoder_hidden_size_2)
        # self.decoder_hidden_size_1 = decoder_hidden_size_1
        # self.decoder_hidden_size_2 = decoder_hidden_size_2
        self.query_size = query_size
        assert query_size == value_size and query_size == key_size, "ERROR: decoder, query_size!=key_size or query_size!=value_size"

        self.query_network = nn.Linear(decoder_hidden_size_2, query_size)

        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.gumbel_noise_weight = 1.0
        # https://casmls.github.io/general/2017/02/01/GumbelSoftmax.html

        self.vocab_size = vocab_size
        self.word2index = word2index
        # print("word2index", word2index)

        self.DEVICE = device

        self.isAttended = isAttended

        if isAttended:
            self.attention = Attention()

        self.character_prob = nn.Linear(decoder_hidden_size_2 + query_size, vocab_size)

    def forward(self, key, values, mask, text=None, isTrain=False, use_gumbel_noise=False,
                return_attention_masks=False):
        '''
        :param key :(T, B, key_size) Output of the Encoder Key projection layer
        :param values: (T, B, value_size) Output of the Encoder Value projection layer
        :param mask: (B, 1, T) be careful, B the batch size is first dim! Useful for attention
        :param text: (N, text_len) Batch input of text with text_length
        :param isTrain: Train or eval mode
        :return predictions: Returns the character perdiction probability
        '''
        batch_size = key.shape[1]
        hidden_size = key.shape[2]

        if (isTrain == True):
            max_len = text.shape[1] - 1  # text: B,T
            # -1 because text: <sos> ... <eos>
            # inputs: <sos>example
            # outputs: example <eos>
            embeddings = self.embedding(text)  # B,T,embed_dim
            # print("dim text", text.shape)
            # print("dim embed", embeddings.size())
        else:
            max_len = 30  # at word-level, 30 words for audio captioning should be enough

        predictions = []
        hidden_states = [None, None]
        prediction = torch.zeros(batch_size, self.vocab_size).to(self.DEVICE)

        # initiating with "<sos>"
        prediction[:, self.word2index['<sos>']] = 1.
        # prediction[:, -1] = 1. # for unit tests

        context = torch.zeros(batch_size, self.query_size).to(
            self.DEVICE)  # initialize context to 0 for the first prediction

        if return_attention_masks: att_masks = []

        for i in range(max_len):

            if (isTrain):
                # Determine if we are using teacher forcing this iteration
                use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
                if i < 1 or use_teacher_forcing:
                    char_embed = embeddings[:, i, :]
                else:
                    if i > 0 and use_gumbel_noise:
                        gumbel_noise = -self.gumbel_noise_weight * torch.log(
                            -torch.log(torch.rand(batch_size, self.vocab_size))).to(self.DEVICE)
                        # print("GUMBEL", gumbel_noise)
                        # pred_wo_gn = prediction.argmax(dim=-1)
                        pred_w_gn = (F_torch.log_softmax(prediction, dim=-1) + gumbel_noise).argmax(dim=-1)
                        # percentage_different_preds = torch.sum(1.*(pred_wo_gn != pred_w_gn))/(pred_wo_gn.size(0))
                        char_embed = self.embedding(pred_w_gn)
                        # print("percentage_different_preds", percentage_different_preds*100)
                    else:
                        char_embed = self.embedding(prediction.argmax(dim=-1))
            else:
                char_embed = self.embedding(prediction.argmax(dim=-1))

            if self.isAttended:
                inp = torch.cat([char_embed, context], dim=1)  # context is size B,h
            elif i < values.size(0):
                inp = torch.cat([char_embed, values[i, :, :]], dim=1)
            else:
                # no more acoustic frames to feed in
                break
            hidden_states[0] = self.lstm1(inp, hidden_states[0])  # outputs a tuple (next_hidden_state, next_cell_state)

            inp_2 = hidden_states[0][0]
            hidden_states[1] = self.lstm2(inp_2, hidden_states[1])

            output = hidden_states[1][0]  # B, hidden//2, ex: B, 256

            if self.isAttended:
                ### Compute attention from the output of the second LSTM Cell ###
                query = self.query_network(output)  # B, h

                # key and values are fixed and are the output of the encoder, size: T,B,h
                context, attention_mask = self.attention(query, key, values,
                                                         mask)  # context: B,h ; attention_mask: B, T_speech_after_pBLSTM_reduction
                context = context.to(self.DEVICE)

                if return_attention_masks:
                    att_masks.append(attention_mask.unsqueeze(1))

                prediction = self.character_prob(torch.cat([output, context],
                                                           dim=1))  # [B, hidden] concat with [B, key_hidden] ---> linear layer -->  B, Vocab
                # print("prediction is on GPU?", prediction.device.index)
            else:
                # if we don't use attention, use values at time step i instead of context
                # not a good idea not using attention though...
                prediction = self.character_prob(torch.cat([output, values[i, :, :]], dim=1))  # B, vocab

            predictions.append(prediction.unsqueeze(1))
        if return_attention_masks:
            return torch.cat(predictions, dim=1).to(self.DEVICE), torch.cat(att_masks, dim=1)
        else:
            return torch.cat(predictions, dim=1).to(self.DEVICE)


class Seq2Seq(nn.Module):
    '''
    wrapper "model" with Encoder-Decoder
    '''

    def __init__(self, input_dim, vocab_size, encoder_hidden_dim=128, use_spec_augment=True, embedding_dim=128, decoder_hidden_size_1=128,
                 decoder_hidden_size_2=128,
                 query_size=128, value_size=128, key_size=128, isAttended=True,
                 pBLSTM_time_reductions=[2, 2, 2],
                 emb_fpath=None, freeze_embeddings=False,
                 teacher_forcing_ratio=0.9, word2index=None, return_attention_masks=False, device='cpu'):

        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_dim, encoder_hidden_dim, value_size, key_size, use_spec_augment=use_spec_augment,
                               pBLSTM_time_reductions=pBLSTM_time_reductions)

        self.decoder = Decoder(vocab_size, embedding_dim=embedding_dim, decoder_hidden_size_1=decoder_hidden_size_1,
                               decoder_hidden_size_2=decoder_hidden_size_2,
                               query_size=query_size, value_size=value_size, key_size=key_size,
                               emb_fpath=emb_fpath, freeze_embeddings=freeze_embeddings,
                               isAttended=isAttended,
                               teacher_forcing_ratio=teacher_forcing_ratio, word2index=word2index, device=device)

        self.pBLSTM_time_reduction_factor = np.prod(pBLSTM_time_reductions)
        self.return_attention_masks = return_attention_masks
        self.DEVICE = device

    def forward(self, audio_input, audio_len, text_input=None, isTrain=True, pretrain_decoder=False, use_gumbel_noise=False,
                return_attention_masks=False):
        """audio_input: a pad sequence sorted by decreasing length"""
        # print("is the model on GPU?", next(self.parameters()).is_cuda)
        # print("audio_input", type(audio_input), audio_input.device.index)

        key, value, out_encoder_lengths = self.encoder(audio_input, audio_len)
        T_after_pBLSTM_reduction, B, _ = key.size()

        if pretrain_decoder:
            # we replace the encoder outputs with random tensors
            key = torch.zeros_like(key, device=self.DEVICE)
            value = torch.zeros_like(value, device=self.DEVICE)
            # key = torch.randn_like(key, device=self.DEVICE)
            # value = torch.randn_like(value, device=self.DEVICE)

        out_encoder_lengths = torch.tensor([l // self.pBLSTM_time_reduction_factor for l in audio_len]).to(self.DEVICE)
        # out_encoder_lengths = torch.tensor([l // (4*self.pBLSTM_time_reduction_factor) for l in audio_len]).to(self.DEVICE)
        out_encoder_lengths = out_encoder_lengths.unsqueeze(1)
        out_encoder_T = audio_len[0] // self.pBLSTM_time_reduction_factor
        # out_encoder_T = audio_len[0] // (4*self.pBLSTM_time_reduction_factor)
        indices = torch.arange(0, out_encoder_T).unsqueeze(0).to(self.DEVICE)
        mask_encoder_output = indices < out_encoder_lengths
        mask_encoder_output = mask_encoder_output.unsqueeze(1).to(self.DEVICE)  # B, 1, T_after_pBLSTM_reduction

        if return_attention_masks:
            predictions, att_masks = self.decoder(key, value, mask_encoder_output, text=text_input, isTrain=isTrain,
                                                  return_attention_masks=return_attention_masks,
                                                  use_gumbel_noise=use_gumbel_noise)
            return predictions, att_masks
        else:
            predictions = self.decoder(key, value, mask_encoder_output, text=text_input, isTrain=isTrain,
                                       return_attention_masks=return_attention_masks, use_gumbel_noise=use_gumbel_noise)

        return predictions  # size: B, T, Vocab


class BeamDecoder(nn.Module):
    """
    Beam search decoder, uses length normalization BS w or w/o LM
    """
    def __init__(self, vocab_size, embedding_dim=128, decoder_hidden_size_1=128, decoder_hidden_size_2=128,
                 query_size=128, value_size=128, key_size=128, isAttended=False, beam_size=10, use_lm_bigram=False, use_lm_trigram=False, lm_weight=0.,
                 teacher_forcing_ratio=0.9, word2index=None, index2word=None, vocab=None, device='cpu'):

        super(BeamDecoder, self).__init__()

        print("BeamDecoder VOCAB SIZE:", vocab_size)

        self.beam_size = beam_size
        self.use_lm_bigram = use_lm_bigram
        self.use_lm_trigram = use_lm_trigram

        if use_lm_bigram or use_lm_trigram:
            self.my_lm = lm_object.ARPALanguageModel(lm_dir + 'dev_pruned.lm')
            # self.my_lm = lm_object.ARPALanguageModel(lm_dir + 'dev.lm')
            # self.my_lm = lm_object.ARPALanguageModel(lm_dir + 'dev_eva.lm')
            self.lm_w = lm_weight

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=word2index['<eos>'])

        self.lstm1 = nn.LSTMCell(input_size=embedding_dim + value_size, hidden_size=decoder_hidden_size_1)
        self.lstm2 = nn.LSTMCell(input_size=decoder_hidden_size_1, hidden_size=decoder_hidden_size_2)
        # self.decoder_hidden_size_1 = decoder_hidden_size_1
        # self.decoder_hidden_size_2 = decoder_hidden_size_2
        self.query_size = query_size

        assert query_size == value_size and query_size == key_size, "ERROR: decoder, query_size!=key_size or query_size!=value_size"

        self.query_network = nn.Linear(decoder_hidden_size_2, query_size)
        # print("hidden size", hidden_dim)
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.gumbel_noise_weight = 1.0
        # https://casmls.github.io/general/2017/02/01/GumbelSoftmax.html

        self.vocab_size = vocab_size
        self.word2index = word2index
        # print("word2index", word2index)
        self.index2word = index2word
        self.vocab = vocab

        self.DEVICE = device

        self.isAttended = isAttended

        if isAttended:
            self.attention = Attention()

        self.character_prob = nn.Linear(decoder_hidden_size_2 + query_size, vocab_size)

    def forward(self, key, values, mask, text=None, isTrain=False, use_gumbel_noise=False,
                return_attention_masks=False):
        '''
        :param key :(T, B, key_size) Output of the Encoder Key projection layer
        :param values: (T, B, value_size) Output of the Encoder Value projection layer
        :param mask: (B, 1, T) be careful, B the batch size is first dim! Useful for attention
        :param text: (N, text_len) Batch input of text with text_length
        :param isTrain: Train or eval mode
        :return predictions: Returns the word prediction probability
        '''

        # TODO: enable batch_size>1

        k = self.beam_size

        # print("key", key.size())

        key = key.expand(-1, k, -1)
        values = values.expand(-1, k, -1)
        mask = mask.expand(k, -1, -1)

        max_output_len = 30  # at word-level, 30 words for audio captioning should be enough

        hidden_states = [None, None]

        # We'll treat the problem as having a batch size of k
        k_prev_words = torch.zeros(k, dtype=torch.long).to(self.DEVICE)

        # initiating with "<sos>"
        k_prev_words[:] = self.word2index['<sos>']
        # prediction[:, -1] = 1. # for unit tests

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(self.DEVICE)  # (k, 1)

        # Tensor to store top k sequences; now they're just <sos>
        seqs = k_prev_words.unsqueeze(1)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        context = torch.zeros(k, self.query_size).to(
            self.DEVICE)  # initialize context to 0 for the first prediction

        hypotheses = list()

        if return_attention_masks: att_masks = []

        for i in range(max_output_len):

            # char_embed = self.embedding(prev_token.argmax(dim=-1))
            char_embed = self.embedding(k_prev_words)

            inp = torch.cat([char_embed, context], dim=1)  # context is size B,h

            hidden_states[0] = self.lstm1(inp, hidden_states[0])  # outputs a tuple (next_hidden_state, next_cell_state)

            inp_2 = hidden_states[0][0]
            hidden_states[1] = self.lstm2(inp_2, hidden_states[1])

            output = hidden_states[1][0]  # B, hidden//2, ex: B, 256

            if self.isAttended:
                ### Compute attention from the output of the second LSTM Cell ###
                query = self.query_network(output)  # B, h
                # key and values are fixed and are the output of the encoder, size: T,B,h

                context, attention_mask = self.attention(query, key, values,
                                                             mask)  # context: B,h ; attention_mask: B, T_speech_after_pBLSTM_reduction
                context = context.to(self.DEVICE)
                if k<2:
                    context = torch.unsqueeze(context, dim=0)
                if return_attention_masks:
                    att_masks.append(attention_mask.unsqueeze(1))

                prediction = self.character_prob(torch.cat([output, context],
                                                           dim=1))  # [B, hidden] concat with [B, key_hidden] ---> linear layer -->  B, Vocab
                # print("prediction is on GPU?", prediction.device.index)
            else:
                # use values at time step i instead of context
                prediction = self.character_prob(torch.cat([output, values[i, :, :]], dim=1))  # B, vocab

            cur_prob = F_torch.log_softmax(prediction, dim=1)

            # Joint ARPA LM decoding
            if self.use_lm_bigram:
                for ind_k in range(k):
                    lm_input = self.index2word[k_prev_words[ind_k].item()]
                    lm_output = torch.zeros(len(self.vocab)).to(self.DEVICE)
                    for ind_w,w in enumerate(self.vocab):
                        lm_output[ind_w] = self.my_lm.scoreword(w, history=(lm_input,))
                    # print(ind_k, "cur_prob", cur_prob[ind_k], "w", self.lm_w, "lm_output", lm_output.log_softmax(dim=-1))
                    cur_prob[ind_k] += self.lm_w * lm_output.log_softmax(dim=-1)

            elif self.use_lm_trigram:
                for ind_k in range(k):
                    current_seq = seqs[ind_k]
                    if len(current_seq) > 1:
                        lm_input = [self.index2word[current_seq[-2].item()],
                                    self.index2word[current_seq[-1].item()]]
                    elif len(current_seq) == 1:
                        lm_input = [self.index2word[0],
                                    self.index2word[current_seq[-1].item()]]
                    else:
                        lm_input = [self.index2word[0]]

                    lm_output = torch.zeros(len(self.vocab)).to(self.DEVICE)
                    for ind_w,w in enumerate(self.vocab):
                        lm_output[ind_w] = self.my_lm.scoreword(w, history=(tuple(lm_input),))
                    # print(ind_k, "cur_prob", cur_prob[ind_k], "w", self.lm_w, "lm_output", lm_output.log_softmax(dim=-1))
                    cur_prob[ind_k] += self.lm_w * lm_output.log_softmax(dim=-1)

            # Add
            scores = top_k_scores.expand_as(cur_prob) + cur_prob  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if i == 0:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / self.vocab_size # (s)
            next_word_inds = top_k_words % self.vocab_size  # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <eos>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != self.word2index['<eos>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                nb_of_words_complete_seqs = [len(caption) for caption in seqs[complete_inds].tolist()]
                raw_scores = top_k_scores[complete_inds]
                # print(nb_of_words_complete_seqs, raw_scores)
                complete_seqs_scores.extend([1. / (nb_of_words_complete_seqs[ind_seq] ** 1.2) * sc for ind_seq, sc in
                                             enumerate(raw_scores.tolist())])
                # complete_seqs_scores.extend(raw_scores.tolist())
                # print(complete_seqs_scores[-len(nb_of_words_complete_seqs):], "\n")

            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            hidden_states[0] = (hidden_states[0][0][prev_word_inds[incomplete_inds]],hidden_states[0][1][prev_word_inds[incomplete_inds]])
            hidden_states[1] = (hidden_states[1][0][prev_word_inds[incomplete_inds]],hidden_states[1][1][prev_word_inds[incomplete_inds]])
            context = context[prev_word_inds[incomplete_inds]]

            key = key[:,:k,:]
            values = values[:,:k,:]
            mask = mask[:k,:,:]

            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds] # .unsqueeze(1)

        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        hypotheses.append([w for w in seq if w not in {self.word2index['<sos>'], self.word2index['<eos>']}])

        # print(complete_seqs_scores[i], hypotheses)
        # no LM
        # tensor(-8.2826)[[1, 107, 32, 1449, 17, 1, 516, 1050]]
        # tensor(-8.8740) [[1, 105, 32, 1119, 17, 1023, 936, 327]]
        # tensor(-8.1964) [[1, 107, 32, 1449, 17, 1, 516, 1050]]

        # LM 2g
        # tensor(-26.1346) [[1, 107, 32, 1449, 17, 1, 516, 1050]]
        # tensor(-24.8034) [[1, 105, 32, 56, 1119, 17, 1023]]
        # tensor(-26.0861) [[1, 107, 32, 1449, 66, 296, 55, 445]]

        # LM 3g
        # tensor(-33.4936) [[1, 107, 32, 1449, 17, 1, 516, 1050]]
        # tensor(-31.4960)[[1, 105, 32, 56, 1119, 17, 1023]]
        # tensor(-33.4074) [[1, 107, 32, 1449, 17, 1, 516, 1050]]

        return hypotheses


class BeamSeq2Seq(nn.Module):
    '''
    wrapper model for the encoder and beam-search decoder.
    '''

    def __init__(self, input_dim, vocab_size, encoder_hidden_dim=128, use_spec_augment=False, embedding_dim=128, decoder_hidden_size_1=128,
                 decoder_hidden_size_2=128,
                 query_size=128, value_size=128, key_size=128, isAttended=True,
                 pBLSTM_time_reductions=[2, 2, 2],
                 teacher_forcing_ratio=0.9, beam_size=2, use_lm_bigram=False, use_lm_trigram=False, lm_weight=0., word2index=None, index2word=None, vocab=None,
                 return_attention_masks=False, device='cpu'):
        super(BeamSeq2Seq, self).__init__()

        self.beam_size = beam_size

        print("beam size:", beam_size)

        self.use_lm_bigram = use_lm_bigram
        self.use_lm_trigram = use_lm_trigram

        self.lm_w = lm_weight

        self.encoder = Encoder(input_dim, encoder_hidden_dim, value_size, key_size, use_spec_augment=use_spec_augment,
                               pBLSTM_time_reductions=pBLSTM_time_reductions)

        # BeamDecoder
        self.decoder = BeamDecoder(vocab_size, embedding_dim=embedding_dim, decoder_hidden_size_1=decoder_hidden_size_1,
                               decoder_hidden_size_2=decoder_hidden_size_2,
                               query_size=query_size, value_size=value_size, key_size=key_size,
                               isAttended=isAttended,
                               use_lm_bigram=use_lm_bigram, use_lm_trigram=use_lm_trigram, lm_weight=lm_weight,
                               beam_size=beam_size,
                               teacher_forcing_ratio=teacher_forcing_ratio, word2index=word2index, index2word=index2word, vocab=vocab,
                                   device=device)

        self.pBLSTM_time_reduction_factor = np.prod(pBLSTM_time_reductions)
        self.return_attention_masks = return_attention_masks
        self.DEVICE = device

    def forward(self, audio_input, audio_len, text_input=None, isTrain=True, use_gumbel_noise=False,
                return_attention_masks=False):
        """audio_input: a pad sequence sorted by decreasing length"""

        # print("audio_input", audio_input.size())
        key, value, out_encoder_lengths = self.encoder(audio_input, audio_len)
        T_after_pBLSTM_reduction, B, _ = key.size()

        out_encoder_lengths = torch.tensor([l // self.pBLSTM_time_reduction_factor for l in audio_len]).to(self.DEVICE)
        out_encoder_lengths = out_encoder_lengths.unsqueeze(1)
        out_encoder_T = audio_len[0] // self.pBLSTM_time_reduction_factor
        indices = torch.arange(0, out_encoder_T).unsqueeze(0).to(self.DEVICE)
        mask_encoder_output = indices < out_encoder_lengths
        mask_encoder_output = mask_encoder_output.unsqueeze(1).to(self.DEVICE)  # B, 1, T_after_pBLSTM_reduction

        beam_predictions = self.decoder(key, value, mask_encoder_output, text=text_input, isTrain=isTrain,
                                       return_attention_masks=return_attention_masks, use_gumbel_noise=use_gumbel_noise)

        return beam_predictions  # size: B, T, Vocab


def masked_ce_loss(probs, targets, lengths, device='cpu'):
    """computes masked CE loss
    param probs: B, L, V
    param targets: B, L
    param lengths: B,
    returns a scalar tensor, averaged over the nb of tokens
    """

    B, L, V = probs.size()

    # count how many tokens we have
    nb_tokens = torch.sum(lengths)

    lengths = lengths.unsqueeze(0)

    indices = torch.arange(0, L).unsqueeze(1).to(device)
    mask = indices < lengths
    assert torch.sum(mask) == nb_tokens, "ERROR: masked_ce_loss, nb of non-zero elements in mask != sum(lengths)"

    probs_flatten = probs.view(-1, V)
    targets_flatten = targets.view(-1)

    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    loss_flat = criterion(probs_flatten, targets_flatten)

    loss_flat = loss_flat.view(L, B)

    masked_loss = mask * loss_flat

    # return masked_loss, nb_tokens
    return torch.sum(masked_loss) / nb_tokens


def masked_ce_loss_per_utt(probs, targets, lengths, device='cpu'):
    """computes masked CE loss per utt
    param probs: B, L, V
    param targets: B, L
    param lengths: B,
    returns a vector with the losses of each recording in a minibatch, normalized by the nb of tokens
    """

    B, L, V = probs.size()

    # count how many tokens we have
    nb_tokens = torch.sum(lengths)

    lengths = lengths.unsqueeze(0)
    # print("lengths", lengths)

    indices = torch.arange(0, L).unsqueeze(1).to(device)
    mask = indices < lengths
    assert torch.sum(mask) == nb_tokens, "ERROR: masked_ce_loss, nb of non-zero elements in mask != sum(lengths)"

    probs_flatten = probs.view(-1, V)
    targets_flatten = targets.view(-1)

    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    loss_flat = criterion(probs_flatten, targets_flatten)

    loss_flat = loss_flat.view(L, B)

    masked_loss = mask * loss_flat

    nb_words_per_utt = torch.sum(mask, dim=0)
    masked_loss_per_utt = torch.sum(masked_loss, dim=0)

    return masked_loss_per_utt / nb_words_per_utt
