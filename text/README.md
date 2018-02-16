## Experiment on Text

Code is based on [Zhao et al., 2017](https://github.com/jakezhaojb/ARAE)

### Requirements
- PyTorch, JSON, Argparse
- (Optional) KenLM (https://github.com/kpu/kenlm)

## Pretrained Version

1) Download and unzip https://drive.google.com/drive/folders/0B4IZ6lmAKTWJSE9UNFYzUkphaVU?usp=sharing.

2) Run: 

    `python generate.py --load_path ./maxlen30`

    (Requires CUDA and Python)



## Data Preparation

### SNLI Data Preparation
- Download dataset and unzip: https://nlp.stanford.edu/projects/snli/snli_1.0.zip
- Run `python snli_preprocessing.py --in_path PATH_TO_SNLI --out_path PATH_TO_PROCESSED_DATA`
    - Example: `python snli_preprocessing.py --in_path ../Data/snli_1.0 --out_path ../Data/snli_lm`
    - The script will create the output directory if it doesn't already exist
- For more information on SNLI see: https://nlp.stanford.edu/projects/snli/

### Your Customized Datasets
If you would like to train a text ARAE on another dataset, simply
1) Create a data directory with a `train.txt` and `test.txt` files with line delimited sentences.
2) Run training command with the `--data_path` argument pointing to that data directory.

## Train
1) To train without KenLM: 

    `python train.py --data_path PATH_TO_PROCESSED_DATA --cuda --no_earlystopping`

2) To train with KenLM for early stopping: 

    `python train.py --data_path PATH_TO_PROCESSED_DATA --cuda --kenlm_path PATH_TO_KENLM_DIRECTORY`

- When training on default parameters the training script will output the logs, generations, and saved models to: `./output/example`

### Model Details
- We train on sentences that have up to 30 tokens and take the most likely word (argmax) when decoding (there is an option to sample when decoding as well).
- For a numerical way for early stopping, after the model has trained for a specified minimum number of epochs, we periodically train a n-gram language model (with modified Kneser-Ney and Laplacian smoothing) on 100,000 generated sentences and evaluate the perplexity of real sentences from a held-out test set. If the perplexity does not improve over that of the lowest perplexity seen for a certain number of iterations (patience), we end training.


### KenLM Installation:
- Download stable release and unzip: http://kheafield.com/code/kenlm.tar.gz
- Need Boost >= 1.42.0 and bjam
    - Ubuntu: `sudo apt-get install libboost-all-dev`
    - Mac: `brew install boost; brew install bjam`
- Run within kenlm directory:
    ```bash
    mkdir -p build
    cd build
    cmake ..
    make -j 4
    ```
- `pip install https://github.com/kpu/kenlm/archive/master.zip`
- For more information on KenLM see: https://github.com/kpu/kenlm and http://kheafield.com/code/kenlm/



