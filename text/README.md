## Experiment on Text

Code is: based on [Zhao et al., 2017](https://github.com/jakezhaojb/ARAE)

### The code for text is not properly ready for use with pre-trained models yet. We will try to get to it soon. ####

If you would still like to create the models at your end you can do so by

0.) Download data files from 
https://drive.google.com/file/d/1afH2kxlFS-B3yxYR4UVNkOLs0sc1NCZl/view?usp=sharing

1.) Training the baseline modeline 
For the baseline model use the files in data/classifier folder
python train_baseline.py --data_path <path> --save_path <model_save_path> <br>

You can also use the ones in the data/classifier/baseline folder instead of training your own

2.) [Optional] Getting the pretrained models for autoencoder, generator and discriminator
https://drive.google.com/file/d/1E1Q5FHf1mUZsz7gPCUDNPsN-ZHKDxaNS/view?usp=sharing

3.) Add DATA_PATH environment variable to the path to store the trained models <br>
export DATA_PATH = <data_path> <br>
For instance you can place the models download above as <data_path>/text_models
 
4.) Train an inverter <br>
For training the inverter use files in data folder <br>
You can either use the pre-trained models in step 2 or train your own models with <br>
python train.py --data_path <path> --update_base --convolution_enc <br>
Once you have the trained models for autoencoder, generator and discriminator, you can train the inverter as below <br>
python train.py --data_path <path> --load_pretrained <dir_name_with_respect_to_model_saved_location> --classifier_path <path>/data/classifier <br>

By default we use fast search and not hybrid search(as in the paper). You can pass an argument --hybrid above to do that

The TreeLSTM and the machine translation results in the paper were done offline and are not included in the respository. If you wish to get those results you can contact ddua@uci.edu for further details


Acknowledgments:
Thanks to jakezhaojb for his excellent code on the ARAE model.
