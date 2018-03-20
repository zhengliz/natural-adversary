## Experiment on Text

Code is: based on [Zhao et al., 2017](https://github.com/jakezhaojb/ARAE)

### The code for text is not properly ready for use with pre-trained models yet. We will try to get to it soon. ####

If you would still like to create the models at your end you can do so by
0.) Download data files from 
https://drive.google.com/file/d/1qOA3P3krBFI6KyfTEPxAmrsaUEC6y-_G/view?usp=sharing

1.) Training the baseline modeline 
python train_baseline.py --data_path <path> --save_path <model_save_path>

2.) [Optional] Getting the pretrained models for autoencoder, generator and discriminator
https://drive.google.com/file/d/14NNZRN1UOB0jLCfcuQHYDMJaG9bfYQwE/view?usp=sharing

3.) Train an inverter
You can either use the pretrained models in step 2 or train your own models with 
python train.py --data_path <path> --update_base --convolution_enc 
Once you have the trained models for autoencoder, generator and discriminator, you can train the inverter as below
python train.py --data_path <path> --load_pretrained <dir_name_with_respect_to_model_saved_location>

The TreeLSTM and the machine translation results in the paper were done offline and are not included in the respository. If you wish to get those results you can contact ddua@uci.edu for further details


Acknowledgments:
Thanks to jakezhaojb for his excellent code on the ARAE model.
