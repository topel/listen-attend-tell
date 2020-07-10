# listen-attend-tell
Audio captioning system based on LAS, used in the DCASE2020 challenge

under construction...

## Installation

1- Setup a virtual env using the environment yaml file:

    conda env create -f environment.yml

Captions are evaluated with metrics used in image captioning. In particular, this module computes the SPIDEr metric, used to rank the systems.

You need to install the COCO caption evaluation module, that has been adapted for the challenge:

    git clone https://github.com/audio-captioning/caption-evaluation-tools

    cd caption-evaluation-tools/coco_caption
    bash ./get_stanford_models.sh
    cd ../..
    mv caption-evaluation-tools/coco_caption .
    mv caption-evaluation-tools/eval_metrics.py .



