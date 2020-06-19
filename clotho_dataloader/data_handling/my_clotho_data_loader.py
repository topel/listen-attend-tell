"""Modified dataloader functions"""

from typing import Callable, Union, Tuple, AnyStr, Optional
from functools import partial
from pathlib import Path

from torch.utils.data.dataloader import DataLoader

from .my_clotho_dataset import ClothoDataset
from .my_collate_fn import clotho_train_collate_fn, clotho_eval_collate_fn

__author__ = 'Konstantinos Drossos'
__docformat__ = 'reStructuredText'
__all__ = ['get_clotho_loader']

def create_dictionaries(unit_list):
    letter2index = dict({el:i for i, el in enumerate(unit_list)})
    index2letter = dict({i:el for i, el in enumerate(unit_list)})
    return letter2index, index2letter


def modify_vocab(unit_list, unit_freq, threshold):
    unit2index = {}
    index2unit = {}
    mapping_index_dict = {}
    new_unit_list = []
    current_index = 0
    for i, el in enumerate(unit_list):
        if unit_freq[i] >= threshold:
            unit2index[el] = current_index
            index2unit[current_index] = el
            mapping_index_dict[i] = current_index
            new_unit_list.append(el)
            current_index += 1

    return unit2index, index2unit, new_unit_list, mapping_index_dict


def get_clotho_loader(data_dir: Path,
                      split: str,
                      input_field_name: str,
                      output_field_name: str,
                      fileid_field_name,
                      load_into_memory: bool,
                      batch_size: int,
                      nb_t_steps_pad: Union[AnyStr, Tuple[int, int]],
                      has_gt_text: bool,

                      shuffle: Optional[bool] = True,
                      drop_last: Optional[bool] = True,
                      input_pad_at: Optional[str] = 'start',
                      output_pad_at: Optional[str] = 'end',
                      mapping_index_dict=None,
                      num_workers: Optional[int] = 1) \
        -> DataLoader:
    """Gets the clotho data loader.

    :param data_dir: Directory with data.
    :type data_dir: pathlib.Path
    :param split: Split to use (i.e. 'development', 'evaluation')
    :type split: str
    :param input_field_name: Field name of the clotho data\
                             to be used as input data to the\
                             method.
    :type input_field_name: str
    :param output_field_name: Field name of the clotho data\
                             to be used as output data to the\
                             method.
    :type output_field_name: str
    :param load_into_memory: Load all data into memory?
    :type load_into_memory: bool
    :param batch_size: Batch size to use.
    :type batch_size: int
    :param nb_t_steps_pad: Number of time steps to\
                           pad/truncate to. Cab use\
                           'max', 'min', or exact number\
                           e.g. (1024, 10).
    :type nb_t_steps_pad: str|(int, int)

    :param has_gt_text: is it the development subset for which we have GT text?
    :type has_gt_text: bool

    :param shuffle: Shuffle examples? Defaults to True.
    :type shuffle: bool, optional
    :param drop_last: Drop the last examples if not making\
                      a batch of `batch_size`? Defaults to True.
    :type drop_last: bool, optional
    :param input_pad_at: Pad input at the start or\
                         at the end?
    :type input_pad_at: str
    :param output_pad_at: Pad output at the start or\
                          at the end?
    :type output_pad_at: str
    :param num_workers: Amount of workers, defaults to 1.
    :type num_workers: int, optional
    :return: Dataloader for Clotho data.
    :rtype: torch.utils.data.dataloader.DataLoader
    """
    dataset: ClothoDataset = ClothoDataset(
        data_dir=data_dir, split=split,
        input_field_name=input_field_name,
        output_field_name=output_field_name,
        fileid_field_name=fileid_field_name,
        load_into_memory=load_into_memory,
        mapping_index_dict=mapping_index_dict,
        has_gt_text=has_gt_text
    )

    print("Number of batches of this dataset:", len(dataset)//batch_size)

    if has_gt_text:
        collate_fn: Callable = partial(
            clotho_train_collate_fn,
            nb_t_steps=nb_t_steps_pad,
            input_pad_at=input_pad_at,
            output_pad_at=output_pad_at)
    else:
        collate_fn: Callable = partial(
            clotho_eval_collate_fn,
            nb_t_steps=nb_t_steps_pad,
            input_pad_at=input_pad_at)

    return DataLoader(
        dataset=dataset, batch_size=batch_size,
        shuffle=shuffle, num_workers=num_workers,
        drop_last=drop_last, collate_fn=collate_fn)

# EOF
