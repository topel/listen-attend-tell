"""Modified collate functions.

Allows to return the audio file ids together with the features and text outputs

"""

from typing import MutableSequence, Union, Tuple, AnyStr
from numpy import ndarray

from torch import cat as pt_cat, zeros as pt_zeros, \
    ones as pt_ones, from_numpy, Tensor, LongTensor

import torch.nn.utils.rnn as rnn_utils

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['clotho_train_collate_fn', 'clotho_eval_collate_fn']


def clotho_train_collate_fn(batch: MutableSequence[ndarray],
                      nb_t_steps: Union[AnyStr, Tuple[int, int]],
                      input_pad_at: str,
                      output_pad_at: str) \
        -> Tuple[Tensor, Tensor, Tensor, Tensor, list]:
    """Pads data.

    :param batch: Batch data.
    :type batch: list[numpy.ndarray]
    :param nb_t_steps: Number of time steps to\
                       pad/truncate to. Cab use\
                       'max', 'min', or exact number\
                       e.g. (1024, 10).
    :type nb_t_steps: str|(int, int)
    :param input_pad_at: Pad input at the start or\
                         at the end?
    :type input_pad_at: str
    :param output_pad_at: Pad output at the start or\
                          at the end?
    :type output_pad_at: str
    :return: Padded data.
    :rtype: torch.Tensor, torch.Tensor
    """
    
    def make_seq_even(sequences, audio_lengths):
        even_seqs = []
        even_len = []
        for i, s in enumerate(sequences):
            if len(s) % 2 != 0:
                even_seqs.append(s[:-1])
                even_len.append(audio_lengths[i]-1)
            else:
                even_seqs.append(s)
                even_len.append(audio_lengths[i])

        return even_seqs, even_len

    if type(nb_t_steps) == str:
        truncate_fn = max if nb_t_steps.lower() == 'max' else min
        in_t_steps = truncate_fn([i[0].shape[0] for i in batch])
        out_t_steps = truncate_fn([i[1].shape[0] for i in batch])
    else:
        in_t_steps, out_t_steps = nb_t_steps

    in_dim = batch[0][0].shape[-1]
    eos_token = batch[0][1][-1]
    
    input_tensor, output_tensor = [], []
    audio_lengths, text_lengths = [], []
    file_ids_list = []

    for in_b, out_b, fileid_b in batch:

        audio_lengths.append(in_b.shape[0])
        # print("toto", out_b.shape)
        text_lengths.append(out_b.shape[0])

        file_ids_list.extend(fileid_b)
        
        if in_t_steps >= in_b.shape[0]:
            padding = pt_zeros(in_t_steps - in_b.shape[0], in_dim).float()
            data = [from_numpy(in_b).float()]
            if input_pad_at.lower() == 'start':
                data.insert(0, padding)
            else:
                data.append(padding)
            tmp_in: Tensor = pt_cat(data)
        else:
            tmp_in: Tensor = from_numpy(in_b[:in_t_steps, :]).float()
        # input_tensor.append(tmp_in.unsqueeze_(0))
        input_tensor.append(tmp_in)

        if out_t_steps >= out_b.shape[0]:
            padding = pt_ones(out_t_steps - len(out_b)).mul(eos_token).long()
            data = [from_numpy(out_b).long()]
            if output_pad_at.lower() == 'start':
                data.insert(0, padding)
            else:
                data.append(padding)

            tmp_out: Tensor = pt_cat(data)
        else:
            tmp_out: Tensor = from_numpy(out_b[:out_t_steps]).long()
        # output_tensor.append(tmp_out.unsqueeze_(0))
        output_tensor.append(tmp_out)

    # we sort by increasing lengths
    # print("audio_lengths", audio_lengths)
    audio_sorted_indices = sorted(range(len(audio_lengths)), key=lambda k: audio_lengths[k])
    audio_batch_sorted = [input_tensor[i] for i in audio_sorted_indices]
    audio_lengths_sorted = [audio_lengths[i] for i in audio_sorted_indices]
    #     print("before, audio_sorted_indices", audio_sorted_indices)
    # print("audio_lengths_sorted", audio_lengths_sorted)

    # get text with the audio_sorted_indices indices
    text_batch_sorted = [output_tensor[i].unsqueeze_(0) for i in audio_sorted_indices]
    text_lengths = [text_lengths[i] for i in audio_sorted_indices]
    # print("text_lengths", text_lengths)
    #     print("before, text_lengths", text_lengths)
    
    # make all audio tensors to even length
    even_audio_batch_sorted, even_audio_lengths_sorted = make_seq_even(audio_batch_sorted, audio_lengths_sorted)

    # reverse lists: largest sequence first (needed for packed sequences)
    # audio_sorted_indices = audio_sorted_indices[::-1]
    even_audio_lengths_sorted = even_audio_lengths_sorted[::-1]
    even_audio_batch_sorted = even_audio_batch_sorted[::-1]

    text_batch_sorted = text_batch_sorted[::-1]
    text_lengths = text_lengths[::-1]

    text_lengths = LongTensor(text_lengths)
    # print("text_lengths tensor", text_lengths)
    text_batch_sorted = pt_cat(text_batch_sorted)
    even_audio_lengths_sorted = LongTensor(even_audio_lengths_sorted)

    # we pad the sequences and get a tensor
    input_tensor = rnn_utils.pad_sequence(even_audio_batch_sorted)  # size: T, B, F=40

    # let's sort the file ids list with the sorted indices:
    # print("????", len(audio_sorted_indices))
    # print("????", len(file_ids_list), file_ids_list)

    file_ids_list_sorted = [file_ids_list[ind] for ind in audio_sorted_indices]
    file_ids_list_sorted = file_ids_list_sorted[::-1]

    # print("????", len(file_ids_list_sorted), file_ids_list_sorted)

    # print('input_tensor', input_tensor.size())
    # print("text_batch_sorted", text_batch_sorted)
    # print("even_audio_lengths_sorted tensor", even_audio_lengths_sorted)
    # print("text_lengths", text_lengths)

    #     print("x_pad", x_pad.size())
    #     for i in range(len(audio_batch)):
    #         print(i, audio_lengths_sorted[i], audio_batch_sorted[i].size(), x_pad[:,i,:].size(), text_lengths[i], padded_text[i].size())

    return input_tensor, text_batch_sorted, even_audio_lengths_sorted, text_lengths, file_ids_list_sorted


def clotho_eval_collate_fn(batch: MutableSequence[ndarray],
                            nb_t_steps: Union[AnyStr, int],
                            input_pad_at: str) \
        -> Tuple[Tensor, Tensor, list]:
    """Pads data.

    :param batch: Batch data.
    :type batch: list[numpy.ndarray]
    :param nb_t_steps: Number of time steps to\
                       pad/truncate to. Cab use\
                       'max', 'min', or exact number\
                       e.g. (1024, 10).
    :type nb_t_steps: str|int
    :param input_pad_at: Pad input at the start or\
                         at the end?
    :type input_pad_at: str
    :return: Padded data.
    :rtype: torch.Tensor, torch.Tensor
    """

    def make_seq_even(sequences, audio_lengths):
        even_seqs = []
        even_len = []
        for i, s in enumerate(sequences):
            if len(s) % 2 != 0:
                even_seqs.append(s[:-1])
                even_len.append(audio_lengths[i] - 1)
            else:
                even_seqs.append(s)
                even_len.append(audio_lengths[i])

        return even_seqs, even_len

    if type(nb_t_steps) == str:
        truncate_fn = max if nb_t_steps.lower() == 'max' else min
        in_t_steps = truncate_fn([i[0][0].shape[0] for i in batch])

    else:
        in_t_steps = nb_t_steps

    in_dim = batch[0][0][0].shape[-1]

    input_tensor= []
    audio_lengths = []
    file_ids_list = []

    for in_b, fileid_b in batch:

        in_b = in_b[0]

        audio_lengths.append(in_b.shape[0])
        # print("toto", out_b.shape)

        file_ids_list.extend(fileid_b)

        if in_t_steps >= in_b.shape[0]:
            padding = pt_zeros(in_t_steps - in_b.shape[0], in_dim).float()
            data = [from_numpy(in_b).float()]
            if input_pad_at.lower() == 'start':
                data.insert(0, padding)
            else:
                data.append(padding)
            tmp_in: Tensor = pt_cat(data)
        else:
            tmp_in: Tensor = from_numpy(in_b[:in_t_steps, :]).float()
        # input_tensor.append(tmp_in.unsqueeze_(0))
        input_tensor.append(tmp_in)

    # we sort by increasing lengths
    # print("audio_lengths", audio_lengths)
    audio_sorted_indices = sorted(range(len(audio_lengths)), key=lambda k: audio_lengths[k])
    audio_batch_sorted = [input_tensor[i] for i in audio_sorted_indices]
    audio_lengths_sorted = [audio_lengths[i] for i in audio_sorted_indices]
    #     print("before, audio_sorted_indices", audio_sorted_indices)
    # print("audio_lengths_sorted", audio_lengths_sorted)

    # make all audio tensors to even length
    even_audio_batch_sorted, even_audio_lengths_sorted = make_seq_even(audio_batch_sorted, audio_lengths_sorted)

    # reverse lists: largest sequence first (needed for packed sequences)
    # audio_sorted_indices = audio_sorted_indices[::-1]
    even_audio_lengths_sorted = even_audio_lengths_sorted[::-1]
    even_audio_batch_sorted = even_audio_batch_sorted[::-1]

    even_audio_lengths_sorted = LongTensor(even_audio_lengths_sorted)

    # we pad the sequences and get a tensor
    input_tensor = rnn_utils.pad_sequence(even_audio_batch_sorted)  # size: T, B, F=40

    # let's sort the file ids list with the sorted indices:

    file_ids_list_sorted = [file_ids_list[ind] for ind in audio_sorted_indices]
    file_ids_list_sorted = file_ids_list_sorted[::-1]

    return input_tensor, even_audio_lengths_sorted, file_ids_list_sorted

# EOF
