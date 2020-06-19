
from typing import Tuple, List, AnyStr, Union
from pathlib import Path

from numpy import ndarray, recarray, array
from torch.utils.data import Dataset
from numpy import load as np_load

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['ClothoDataset']


# In [13]: toto = np.load("../../dcase2020/task6/clotho-dataset/data/clotho_dataset_eva/clotho_file_01 A pug struggles to breathe 1_14_2008.wav_0.npy", allow_pickle=True)

# In [14]: toto
# Out[14]:
# rec.array([('01 A pug struggles to breathe 1_14_2008.wav', array([[-2.703846 , -2.5893888, -3.526077 , ..., -8.308511 , -8.478548 ,
#         -8.9394045],
#        [-2.3663898, -3.1732218, -4.541931 , ..., -7.26185  , -7.1988964,
#         -7.55435  ],
#        [-2.0927114, -2.733207 , -4.5192285, ..., -5.7005363, -6.0805564,
#         -6.915228 ],
#        ...,
#        [-1.7230142, -1.8596828, -3.3882952, ..., -7.7823725, -8.028824 ,
#         -8.390904 ],
#        [-1.3847176, -1.699652 , -2.6164887, ..., -7.858222 , -8.049089 ,
#         -8.228726 ],
#        [-1.8532189, -1.9424245, -2.9929178, ..., -8.021147 , -8.057826 ,
#         -8.090835 ]], dtype=float32), '<SOS> A man walking who is blowing his nose hard and about to sneeze. <EOS>', 0, array([   0,    1,  516,  559,  301,   32,  285,  290, 2834,  440,   17,
#        1048,   51, 3666,    9]), array([31,  1,  0,  1,  2,  0,  7,  1, 24,  0,  5, 14,  9,  7, 23,  1, 24,
#        16,  8,  1,  9, 10,  1, 12,  5,  8, 24,  9,  7, 23,  1, 16,  9, 10,
#         1,  7,  8, 10,  6,  1, 16,  0, 13,  4,  1,  0,  7,  4,  1,  0, 12,
#         8,  3, 17,  1, 17,  8,  1, 10,  7,  6,  6, 22,  6, 20,  1, 32]))],
#           dtype=[('file_name', '<U43'), ('features', 'O'), ('caption', '<U75'), ('caption_ind', '<i4'), ('words_ind', 'O'), ('chars_ind', 'O')])


class ClothoDataset(Dataset):
    def __init__(self, data_dir: Path,
                 split: AnyStr,
                 input_field_name: AnyStr,
                 output_field_name: AnyStr,
                 fileid_field_name,
                 load_into_memory: bool,
                 mapping_index_dict,
                 has_gt_text: bool) \
            -> None:
        """Initialization of a Clotho dataset object.

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
        :param return_file_id: whether to return file id
        :type return_file_id: bool
        :param has_gt_text: is it the development subset for which we have GT text?
        :type has_gt_text: bool


        """
        super(ClothoDataset, self).__init__()
        the_dir: Path = data_dir.joinpath(split)

        self.examples: List[Path] = sorted(the_dir.iterdir())
        self.input_name: str = input_field_name
        self.output_name: str = output_field_name
        self.load_into_memory: bool = load_into_memory
        self.fileid_field_name = fileid_field_name
        self.mapping_index_dict = mapping_index_dict
        self.has_gt_text: bool = has_gt_text

        if load_into_memory:
            self.examples: List[recarray] = [np_load(str(f), allow_pickle=True)
                                             for f in self.examples]

    def __len__(self) \
            -> int:
        """Gets the amount of examples in the dataset.

        :return: Amount of examples in the dataset.
        :rtype: int
        """
        return len(self.examples)

    def __getitem__(self,
                    item: int) \
            -> Tuple[ndarray, ndarray]:
        """Gets an example from the dataset.

        :param item: Index of the item.
        :type item: int
        :return: Input and output values.
        :rtype: numpy.ndarray. numpy.ndarray
        """
        ex: Union[Path, recarray] = self.examples[item]
        if not self.load_into_memory:
            ex: recarray = np_load(str(ex), allow_pickle=True)
        if self.fileid_field_name is not None:
            fid_e = [ex[i].item() for i in [self.fileid_field_name]]
        if self.has_gt_text:
            in_e, ou_e = [ex[i].item() for i in [self.input_name, self.output_name]]
            # print("dataset class, text with 5k words:", ou_e, len(ou_e))
            if self.mapping_index_dict is not None:
                ou_e = [self.mapping_index_dict[ind] for ind in ou_e if ind in self.mapping_index_dict]
                ou_e = array(ou_e, dtype=int)
                # print("               text with 1k words:", ou_e, len(ou_e))
            if self.fileid_field_name is not None:
                return in_e, ou_e, fid_e
            else:
                return in_e, ou_e

        else:
            in_e = [ex[i].item() for i in [self.input_name]]
            if self.fileid_field_name is not None:
                return in_e, fid_e
            else:
                return in_e

