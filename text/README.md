## Experiment on Text

0.) Data files are in `./data` and baseline models are in `./models/baseline`

1.) [Optional] Train the baseline classifier models for textual entailment: TE-embeddings and TE-lstm. 

`python train_baseline.py --data_path ./data --save_path ./models/baseline` 

You can also use the ones in `./models/baseline` instead of training your own.

2.) Train the autoencoder (ARAE), generator and discriminator (GAN)

`python train.py --data_path ./data --update_base --convolution_enc --classifier_path ./models`
 
3.) Train the inverter

Once you have pretrained models for autoencoder, generator and discriminator, you can train the inverter as below

`python train.py --data_path ./data --load_pretrained <pretrain_exp_ID> --classifier_path ./models`

4.) By default, we use fast search instead of hybrid search (as described in the paper). 

You can pass an argument `--hybrid` above to change that.

5.) TE-treeLSTM and machine translation results in the paper were done offline and are not included in the respository. 

If you wish to get those results, you can contact `ddua@uci.edu` for further details.

#### Acknowledgment
Initial code is based on [Zhao et al., 2017](https://github.com/jakezhaojb/ARAE)
