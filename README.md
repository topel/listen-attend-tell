# listen-attend-tell
Audio captioning system based on LAS, used in the DCASE2020 challenge

If you use this code, please consider citing the tech report:

    @techreport{pellegrini2020_t6,
        Author = "Pellegrini, Thomas",
        title = "{IRIT}-{UPS} {DCASE} 2020 audio captioning system",
        institution = "DCASE2020 Challenge",
        year = "2020",
        month = "June",
        abstract = "This technical report is a short description of the sequence-to-sequence model used in the DCASE 2020 task 6 dedicated to audio captioning. Four submissions were made: i) a baseline one using greedy search, ii) beam search, iii) beam search integrating a 2g language model, iv) with a model trained with a vocabulary limited to the most frequent word types (1k words instead of about 5k words)."
    }

## Installation

0- Clone this directory

1- Setup a virtual env using the environment yaml file:

    conda env create -f environment.yml

2- Captions are evaluated with metrics used in image captioning. In particular, this module computes the SPIDEr metric, used to rank the systems.

You need to install the COCO caption evaluation module, that has been adapted for the challenge:

    git clone https://github.com/audio-captioning/caption-evaluation-tools

    cd caption-evaluation-tools/coco_caption
    bash ./get_stanford_models.sh
    cd ../..
    mv caption-evaluation-tools/coco_caption .
    mv caption-evaluation-tools/eval_metrics.py .
 
## Download the Clotho dataset and extract the log F-BANK coefficients

For instructions, see http://dcase.community/challenge2020/task-automatic-audio-captioning

Once you downloaded the dataset, you might put one directory above this repository. My directory structure looks like:

    - dcase2020_task6/
      |
      - listen-attend-tell: this source code directory
      - clotho-dataset
        |
        - data: where all the .csv files, .WAV and .npy dirs are  
        - lm
    
You can of course change this, but you will need to edit the data_dir variable in the main_\*.py scripts.

## Train a model

You can train a model with the following call:

    python main_train.py 0.98 2 2
    
where:
- "0.98" is the teacher-forcing probability threshold: there are 98% chance to apply teacher forcing 
- "2 2" tells the script to use two stacked pyramidal BLSTM layers, each with a time reduction factor of 2. This factor can be 2, 4 or 8. You can use between 1 and 3 layers. for instance, "4 8 8" means you want three layers with time reduction factors of 4, 8, 8, respectively. 

## Download a pre-trained network

The model I used to submit predictions is available: 
- https://zenodo.org/record/3893974#.XxB3sB9fjCI


## Test a model

You can test a trained model on the development-testing subset and get caption metrics, with the following call:

    python main_decode.py 0.98 2 2
    
By default, decoding is done with greedy search. You can do beam search by modifying these two lines:

    do_decode_val = True
    do_decode_val_beamsearch = False

by

    do_decode_val = False
    do_decode_val_beamsearch = true
    
and edit parameters, such as beam size and if you want to use a language model or not.    

