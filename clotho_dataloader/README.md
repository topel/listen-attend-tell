# Clotho data handling

Welcome to Clotho data handling repository. This repository has the necessary code for
using the DataLoader class from PyTorch package (`torch.utils.data.dataloader.DataLoader`)
with the Clotho dataset. 

You can use the present data loader of Clotho directly with the examples created by the
[Clotho baseline dataset repository](https://github.com/dr-costas/clotho-baseline-dataset). 

If you are looking at this README file, then I suppose that you already know what is a
DataLoader from PyTorch. Nevertheless, the Clotho dataset has sequences as inputs and outputs,
and each sequence is of arbitrary length (15 to 30 seconds for the input and 8 to 20 words 
for the output). For that reason, this data loader already provides a collate function. 

This repository is maintained by [K. Drossos](https://github.com/dr-costas).


----


## Clotho dataset class

In the `data_handling` package, there is the `clotho_dataset.py`, which holds the `ClothoDataset` 
class. This class offers the functionality of a `PyTorch` dataset object, tuned for the Clotho 
dataset. 

The `ClothoDataset` object needs the following arguments: 

 - `data_dir` which is the directory that has the data of the Clotho dataset (i.e. the root
   directory of the Clotho dataset). This argument should be of type `pathlib.Path`. 
 - `spilt` which is the split that you want to use, and the argument should be of type `str`.
 - `input_field_name` which is the field name of the `numpy.recarray` that holds the input data
   to your audio captioning method. Currently, only single input fields are supported (i.e. you
   cannot specify multiple fields). The type of this argument should be `str`. 
 - `output_field_name` is the ield name of the `numpy.recarray` that holds the output data
   to your audio captioning method. Currently, only single output fields are supported (i.e. you
   cannot specify multiple fields). The type of this argument should be `str`.
 - `load_into_memory` which is a `bool` flag for indicating if the data in the dataset should be
   loaded into memory or read from the disk when needed.
    
    
----
    
## Clotho data loader

The data loader is just a function, wrapping the creation of a `torch.utils.data.DataLoader` class, 
that also offers functionality for instantiating the `ClothoDataset` class and the collate function, 
that will be used with the data loader. 

The data loader of Clotho needs the following arguments: 

 - `data_dir` which is the directory that has the data of the Clotho dataset (i.e. the root
    directory of the Clotho dataset). This argument should be of type `pathlib.Path`. 
 - `spilt` which is the split that you want to use, and the argument should be of type `str`.
 - `input_field_name` which is the field name of the `numpy.recarray` that holds the input data
    to your audio captioning method. Currently, only single input fields are supported (i.e. you
    cannot specify multiple fields). The type of this argument should be `str`. 
 - `output_field_name` is the ield name of the `numpy.recarray` that holds the output data
    to your audio captioning method. Currently, only single output fields are supported (i.e. you
    cannot specify multiple fields). The type of this argument should be `str`.
 - `load_into_memory` which is a `bool` flag for indicating if the data in the dataset should be
    loaded into memory or read from the disk when needed.  
 - `batch_size` is the batch size to be used with the data loader. This argument should be an `int`. 
 - `nb_t_steps_pad` is the number of time-steps to pad or truncate the sequences using the collate
   function. This argument can be an `int` (i.e. the actual time-steps) but also can be the strings
   `max` or `min`, meaning pad/truncate to maximum/minimum amount of time-steps in the batch. 
   Currently, zeros (input audio) and <EOF> tokens (output words) are supported for padding.  
   the padding. Supported values for `str` are `max` and `min`.
 - `shuffle` flag to indicate the shuffling of the data, exactly as in the `torch.utils.data.DataLoader`
   class. This argument should be a `bool`.
 - `drop_last` flag to indicate the dropping of the examples that cannot grouped in a batch,  
   exactly as in the `torch.utils.data.DataLoader` class. This argument should be a `bool`.
 - `input_pad_at` where to pad the input sequence at, i.e. at the `start` or at the `end`? This
   argument should be a `str` and supported strings are `start` and `end`.
 - `output_pad_at` the same as `input_pad_at`, but for the output sequence. 
 - `num_workers` is the amount of workers to be used for the data loader, exactly as in 
   the `torch.utils.data.DataLoader` class. This argument should be an `int`. 


----


## Collate function

To be able to use the sequences of Clotho in a batch, you most likely will need some kind of padding
policy. This repository already offers a collate function to be used with the Clotho data. 

With the provided collate function, you can choose to either: 

* pad the data with zeros (for input audio data) and end-of-sequence symbol (for the output/words), 
to the length of the longest input (for the inputs) and output (for the outputs) sequence in
tha batch
* truncate the input and the output to the minimum length of the input and output in the batch, and
* use a constant length for input and output, and either truncate or pad. 

Enjoy and if you have any issues, please let me know in the issue section. 
