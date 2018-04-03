import argparse
import os
import time
import math
import numpy as np
import random
import sys
import json
import torch.utils.data
import pickle as pkl
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from search import search, search_fast
from utils import to_gpu, Corpus, batchify, SNLIDataset, train_ngram_lm, get_ppl, load_ngram_lm, get_delta, collate_snli
from models import Seq2Seq, MLP_D, MLP_G, MLP_I, MLP_I_AE, JSDistance, Seq2SeqCAE, Baseline_Embeddings, Baseline_LSTM


parser = argparse.ArgumentParser(description='PyTorch ARAE for Text')
# Path Arguments
parser.add_argument('--data_path', type=str, required=True,
                    help='location of the data corpus')
parser.add_argument('--classifier_path', type=str, required=True,
                    help='location of the classifier files')
parser.add_argument('--kenlm_path', type=str, default='/home/ddua/kenlm',
                    help='path to kenlm directory')
parser.add_argument('--outf', type=str, default='example',
                    help='output directory name')

# Data Processing Arguments
parser.add_argument('--vocab_size', type=int, default=11000,
                    help='cut vocabulary down to this size '
                         '(most frequently seen words in train)')
parser.add_argument('--maxlen', type=int, default=10,
                    help='maximum sentence length')
parser.add_argument('--lowercase', type=bool, default=True,
                    help='lowercase all text')
parser.add_argument('--packed_rep', type=bool, default=False,
                    help='pad all sentences to fixed maxlen')


# Model Arguments
parser.add_argument('--emsize', type=int, default=300,
                    help='size of word embeddings')
parser.add_argument('--nhidden', type=int, default=300,
                    help='number of hidden units per layer in LSTM')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--noise_radius', type=float, default=0.2,
                    help='stdev of noise for autoencoder (regularizer)')
parser.add_argument('--noise_anneal', type=float, default=0.995,
                    help='anneal noise_radius exponentially by this'
                         'every 100 iterations')
parser.add_argument('--hidden_init', action='store_true',
                    help="initialize decoder hidden state with encoder's")
parser.add_argument('--arch_i', type=str, default='300-300',
                    help='inverter architecture (MLP)')
parser.add_argument('--arch_g', type=str, default='300-300',
                    help='generator architecture (MLP)')
parser.add_argument('--arch_d', type=str, default='300-300',
                    help='critic/discriminator architecture (MLP)')
parser.add_argument('--arch_conv_filters', type=str, default='500-700-1000',
                    help='encoder filter sizes for different convolutional layers')
parser.add_argument('--arch_conv_strides', type=str, default='1-2-2',
                    help='encoder strides for different convolutional layers')
parser.add_argument('--arch_conv_windows', type=str, default='3-3-3',
                    help='encoder window sizes for different convolutional layers')
parser.add_argument('--z_size', type=int, default=100,
                    help='dimension of random noise z to feed into generator')
parser.add_argument('--temp', type=float, default=1,
                    help='softmax temperature (lower --> more discrete)')
parser.add_argument('--enc_grad_norm', type=bool, default=True,
                    help='norm code gradient from critic->encoder')
parser.add_argument('--gan_toenc', type=float, default=-0.01,
                    help='weight factor passing gradient from gan to encoder')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--useJS', type=bool, default=True,
                    help='use Jenson Shannon distance')
parser.add_argument('--perturb_z', type=bool, default=True,
                    help='perturb noise space z instead of hidden c')


# Training Arguments
parser.add_argument('--epochs', type=int, default=15,
                    help='maximum number of epochs')
parser.add_argument('--min_epochs', type=int, default=20,
                    help="minimum number of epochs to train for")
parser.add_argument('--no_earlystopping', action='store_true',
                    help="won't use KenLM for early stopping")
parser.add_argument('--patience', type=int, default=5,
                    help="number of language model evaluations without ppl "
                         "improvement to wait before early stopping")
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--niters_ae', type=int, default=1,
                    help='number of autoencoder iterations in training')
parser.add_argument('--niters_gan_d', type=int, default=5,
                    help='number of discriminator iterations in training')
parser.add_argument('--niters_gan_g', type=int, default=1,
                    help='number of generator iterations in training')
parser.add_argument('--niters_inv', type=int, default=5,
                    help='number of inverter iterations in training')
parser.add_argument('--niters_gan_schedule', type=str, default='2-4-6',
                    help='epoch counts to increase number of GAN training '
                         ' iterations (increment by 1 each time)')
parser.add_argument('--lr_ae', type=float, default=1,
                    help='autoencoder learning rate')
parser.add_argument('--lr_inv', type=float, default=1e-05,
                    help='inverter learning rate')
parser.add_argument('--lr_gan_g', type=float, default=5e-05,
                    help='generator learning rate')
parser.add_argument('--lr_gan_d', type=float, default=1e-05,
                    help='critic/discriminator learning rate')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='beta1 for adam. default=0.9')
parser.add_argument('--clip', type=float, default=1,
                    help='gradient clipping, max norm')
parser.add_argument('--gan_clamp', type=float, default=0.01,
                    help='WGAN clamp')
parser.add_argument('--convolution_enc', action='store_true', default=False,
                    help='use convolutions in encoder')
parser.add_argument('--use_inv_ae', action='store_true', default=False,
                    help='use encoder->inv->gen->dec')
parser.add_argument('--update_base', action='store_true', default=False,
                    help='updating base models')
parser.add_argument('--reload_exp', type=str, default=None,
                    help='resume a previous experiment')
parser.add_argument('--load_pretrained', type=str, default=None,
                    help='load a pre-trained encoder and decoder to train the inverter')

# Evaluation Arguments
parser.add_argument('--sample', action='store_true',
                    help='sample when decoding for generation')
parser.add_argument('--N', type=int, default=5,
                    help='N-gram order for training n-gram language model')
parser.add_argument('--log_interval', type=int, default=200,
                    help='interval to log autoencoder training results')


# Other
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='use CUDA')
parser.add_argument('--debug_mode', action='store_true', default=False,
                    help='debug mode to not create a new dir')
parser.add_argument('--hybrid', type=bool, default=False,
                    help='performs hybrid search')

args = parser.parse_args()
print(vars(args))


if args.cuda:
    torch.cuda.device(0)
    print(torch.cuda.device_count())
    print("setting cuda device to "+format(torch.cuda.current_device()))
if args.debug_mode:
    args.outf = args.outf+"/debug"
elif args.reload_exp:
    args.outf = args.outf+"/"+args.reload_exp
else:
    args.outf = args.outf+"/"+str(int(time.time()))
    
# make output directory if it doesn't already exist
if not os.path.isdir(os.environ["DATA_PATH"]+'/arae/output'):
    os.makedirs(os.environ["DATA_PATH"]+'/arae/output')
if not os.path.isdir(os.environ["DATA_PATH"]+'/arae/output/{}'.format(args.outf)):
    os.makedirs(os.environ["DATA_PATH"]+'/arae/output/{}'.format(args.outf))
    os.makedirs(os.environ["DATA_PATH"]+'/arae/output/{}'.format(args.outf+"/models"))
print("Dumping into directory {0}".format(args.outf))


# Set the random seed manually for reproducibility.
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, "
              "so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)


if args.reload_exp:
    cur_dir = os.environ["DATA_PATH"]+'/arae/output/example/{}/'.format(args.reload_exp)
    print("Loading experiment from "+ cur_dir)
elif args.load_pretrained:
    cur_dir = os.environ["DATA_PATH"]+'/arae/output/example/{}/'.format(args.load_pretrained)
    print("Loading experiment from "+ cur_dir)
else:
    cur_dir = os.environ["DATA_PATH"]+'/arae/output/example/'.format(args.outf)
    print("Create experiment at "+ cur_dir)
    
###############################################################################
# Load data
###############################################################################

# create corpus
if args.load_pretrained:
    corpus = Corpus(args.data_path,
                maxlen=args.maxlen,
                vocab_size=args.vocab_size,
                lowercase=args.lowercase,
                load_vocab=cur_dir+'/vocab.json')
else:
    corpus = Corpus(args.data_path,
                maxlen=args.maxlen,
                vocab_size=args.vocab_size,
                lowercase=args.lowercase)
    
eval_batch_size = 10
if not args.convolution_enc:
    args.packed_rep = True
train_data = batchify(corpus.train, args.batch_size, args.maxlen, packed_rep=args.packed_rep, shuffle=True)
corpus_test = SNLIDataset(train=False, vocab_size=args.vocab_size+4, reset_vocab=corpus.dictionary.word2idx)
testloader = torch.utils.data.DataLoader(corpus_test, batch_size=10, collate_fn=collate_snli, shuffle=False)
test_data = iter(testloader)

classifier1 = Baseline_Embeddings(100,vocab_size=args.vocab_size+4)
classifier1.load_state_dict(torch.load(args.classifier_path+"/baseline/model_emb.pt"))
classifier2 = Baseline_LSTM(100,300,maxlen=10, gpu=args.cuda)
classifier2.load_state_dict(torch.load(args.classifier_path+"/baseline/model_lstm.pt"))

vocab_classifier1 = pkl.load(open(args.classifier_path+"/vocab.pkl", 'r'))
vocab_classifier2 = pkl.load(open(args.classifier_path+"/vocab.pkl", 'r'))

print("Loaded data!")
    
ntokens = len(corpus.dictionary.word2idx)
start_epoch = 1

if args.reload_exp:
    file_numbers = [int(f.split("_")[0]) for f in os.listdir(cur_dir) if "_perturbation" in f]
    file_numbers.sort()
    start_epoch = file_numbers[-1]+1
    print("Starting with epoch :{0}".format(start_epoch))

###############################################################################
# Build the models
###############################################################################

if args.reload_exp or args.load_pretrained:
    autoencoder = torch.load(open(cur_dir+'/models/autoencoder_model.pt'))
    gan_gen = torch.load(open(cur_dir+'/models/gan_gen_model.pt'))
    gan_disc = torch.load(open(cur_dir+'/models/gan_disc_model.pt'))
    with open(cur_dir+'/vocab.json', 'r') as f:
        corpus.dictionary.word2idx = json.load(f)

    if args.load_pretrained:
        inverter = MLP_I(args.nhidden, args.z_size, args.arch_i, gpu=args.cuda)
    else:
        inverter = torch.load(open(cur_dir+'/models/inverter_model.pt'))
else:
    if args.convolution_enc:
        autoencoder = Seq2SeqCAE(emsize=args.emsize,
                              nhidden=args.nhidden,
                              ntokens=ntokens,
                              nlayers=args.nlayers,
                              noise_radius=args.noise_radius,
                              hidden_init=args.hidden_init,
                              dropout=args.dropout,
                              conv_layer=args.arch_conv_filters,
                              conv_windows=args.arch_conv_windows,
                              conv_strides=args.arch_conv_strides,
                              gpu=args.cuda)
    else:
        autoencoder = Seq2Seq(emsize=args.emsize,
                              nhidden=args.nhidden,
                              ntokens=ntokens,
                              nlayers=args.nlayers,
                              noise_radius=args.noise_radius,
                              hidden_init=args.hidden_init,
                              dropout=args.dropout,
                              gpu=args.cuda)
    inverter = MLP_I_AE(ninput=args.nhidden, noutput=args.z_size, layers=args.arch_i)
    gan_gen = MLP_G(ninput=args.z_size, noutput=args.nhidden, layers=args.arch_g)
    gan_disc = MLP_D(ninput=args.nhidden, noutput=1, layers=args.arch_d)
    # dumping vocabulary
    with open(os.environ["DATA_PATH"]+'/arae/output/{}/vocab.json'.format(args.outf), 'w') as f:
        json.dump(corpus.dictionary.word2idx, f, encoding='utf-8')


if os.path.exists(os.environ["DATA_PATH"]+'/arae/output/{}/args.json'.format(args.outf)):
    with open(os.environ["DATA_PATH"]+'/arae/output/{}/args_2.json'.format(args.outf), 'w') as f:
        json.dump(vars(args), f)
    with open(os.environ["DATA_PATH"]+'/arae/output/{}/logs.txt'.format(args.outf), 'a') as f:
        f.write(str(vars(args)))
        f.write("\n\n")
        f.write("Loading experiment from "+ cur_dir)
        f.write("\n")
        f.write("Starting with epoch :{0}\n".format(start_epoch))
else:
    with open(os.environ["DATA_PATH"]+'/arae/output/{}/args.json'.format(args.outf), 'w') as f:
        json.dump(vars(args), f)
    with open(os.environ["DATA_PATH"]+'/arae/output/{}/logs.txt'.format(args.outf), 'w') as f:
        f.write(str(vars(args)))
        f.write("\n")
        f.write("Create experiment at "+ cur_dir)
        f.write("\n")
        f.write("Starting with epoch :{0}\n".format(start_epoch))

ntokens = len(corpus.dictionary.word2idx)
print("Vocabulary Size: {}".format(ntokens))
args.ntokens = ntokens


print(autoencoder)
print(inverter)
print(gan_gen)
print(gan_disc)

optimizer_ae = optim.SGD(autoencoder.parameters(), lr=args.lr_ae)
optimizer_inv = optim.Adam(inverter.parameters(),
                           lr=args.lr_inv,
                           betas=(args.beta1, 0.999))
optimizer_gan_g = optim.Adam(gan_gen.parameters(),
                             lr=args.lr_gan_g,
                             betas=(args.beta1, 0.999))
optimizer_gan_d = optim.Adam(gan_disc.parameters(),
                             lr=args.lr_gan_d,
                             betas=(args.beta1, 0.999))

criterion_ce = nn.CrossEntropyLoss()
criterion_mse = nn.MSELoss()
criterion_js = JSDistance()


if args.cuda:
    autoencoder = autoencoder.cuda()
    inverter = inverter.cuda()
    gan_gen = gan_gen.cuda()
    gan_disc = gan_disc.cuda()
    criterion_ce = criterion_ce.cuda()
    classifier1 = classifier1.cuda()
    classifier2 = classifier2.cuda()
else:
    autoencoder.gpu = False
    autoencoder = autoencoder.cpu()
    gan_gen = gan_gen.cpu()
    gan_disc = gan_disc.cpu()
    inverter = inverter.cpu()
    classifier1.cpu()
    classifier2.cpu()

###############################################################################
# Training code
###############################################################################


def pred_fn(data):
    gpu = args.cuda
    premise, hyp_indices, hypothesis_c, dist = data
    edit_dist = []
    premise_words = " ".join([corpus_test.dictionary.idx2word[x] for x in premise.data.cpu().numpy()[0]])
    premise_words_indices1 = [vocab_classifier1[w] if w in vocab_classifier1 else 3 for w in premise_words.strip().split()]
    premise_words_indices1 = Variable(torch.LongTensor(premise_words_indices1)).unsqueeze(0)
    
    premise_words_indices2 = [vocab_classifier2[w] if w in vocab_classifier2 else 3 for w in premise_words.strip().split()]
    premise_words_indices2 = Variable(torch.LongTensor(premise_words_indices2)).unsqueeze(0)
    
    hyp_sample_idx = autoencoder.generate(hypothesis_c, 10, True).data.cpu().numpy()
    words_all = []
    premise_word_inds1 = []
    premise_word_inds2 = []
    hypothesis_word_inds1 = []
    hypothesis_word_inds2 = []
    for i in range(hyp_sample_idx.shape[0]):
        words = [corpus_test.dictionary.idx2word[x] for x in hyp_sample_idx[i]]
        words_all.append(" ".join(words)+"\t"+str(dist[i]))
        
        edit_dist.append(len(set(hyp_indices[0].data.cpu().numpy()).intersection(set(hyp_sample_idx[0]))))
        hypothesis_word_indx1 = [vocab_classifier1[w] if w in vocab_classifier1 else 3 for w in words]
        hypothesis_word_indx1 = Variable(torch.LongTensor(hypothesis_word_indx1)).unsqueeze(0)
        hypothesis_word_indx2 = [vocab_classifier2[w] if w in vocab_classifier2 else 3 for w in words]
        hypothesis_word_indx2 = Variable(torch.LongTensor(hypothesis_word_indx2)).unsqueeze(0)
        if gpu:
            premise_words_indices1 = premise_words_indices1.cuda()
            premise_words_indices2 = premise_words_indices2.cuda()
            hypothesis_word_indx1 = hypothesis_word_indx1.cuda()
            hypothesis_word_indx2 = hypothesis_word_indx2.cuda()
            
        premise_word_inds1.append(premise_words_indices1)
        premise_word_inds2.append(premise_words_indices2)
        hypothesis_word_inds1.append(hypothesis_word_indx1)
        hypothesis_word_inds2.append(hypothesis_word_indx2)
        
    premise_word_inds1 = torch.cat(premise_word_inds1, 0)
    premise_word_inds2 = torch.cat(premise_word_inds2, 0)
    hypothesis_word_inds1 = torch.cat(hypothesis_word_inds1, 0)
    hypothesis_word_inds2 = torch.cat(hypothesis_word_inds2, 0)

    prob_distrib1 = classifier1((premise_word_inds1, hypothesis_word_inds1))
    prob_distrib2 = classifier2((premise_word_inds2, hypothesis_word_inds2))
        
    _, predictions1 = torch.max(prob_distrib1, 1)
    _, predictions2 = torch.max(prob_distrib2, 1)
    return predictions1, predictions2, words_all


def save_model(epoch=None):
  if not epoch and args.load_pretrained:
    with open(os.environ["DATA_PATH"]+'/arae/output/{0}/{1}/inverter_model.pt'.format(args.outf+"/models",epoch), 'wb') as f:
      torch.save(inverter, f)
    return
  if epoch and epoch%5==0:
    print("Saving models")
    if not os.path.isdir(os.environ["DATA_PATH"]+'/arae/output/{}'.format(epoch)):
      os.makedirs(os.environ["DATA_PATH"]+'/arae/output/{0}/{1}'.format(args.outf+"/models",epoch))
    with open(os.environ["DATA_PATH"]+'/arae/output/{0}/{1}/autoencoder_model.pt'.format(args.outf+"/models",epoch), 'wb') as f:
      torch.save(autoencoder, f)
    with open(os.environ["DATA_PATH"]+'/arae/output/{0}/{1}/inverter_model.pt'.format(args.outf+"/models",epoch), 'wb') as f:
      torch.save(inverter, f)
    with open(os.environ["DATA_PATH"]+'/arae/output/{0}/{1}/gan_gen_model.pt'.format(args.outf+"/models",epoch), 'wb') as f:
      torch.save(gan_gen, f)
    with open(os.environ["DATA_PATH"]+'/arae/output/{0}/{1}/gan_disc_model.pt'.format(args.outf+"/models",epoch), 'wb') as f:
      torch.save(gan_disc, f)
  else:
    print("Saving models")
    with open(os.environ["DATA_PATH"]+'/arae/output/{0}/autoencoder_model.pt'.format(args.outf+"/models"), 'wb') as f:
      torch.save(autoencoder, f)
    with open(os.environ["DATA_PATH"]+'/arae/output/{0}/inverter_model.pt'.format(args.outf+"/models"), 'wb') as f:
      torch.save(inverter, f)
    with open(os.environ["DATA_PATH"]+'/arae/output/{0}/gan_gen_model.pt'.format(args.outf+"/models"), 'wb') as f:
      torch.save(gan_gen, f)
    with open(os.environ["DATA_PATH"]+'/arae/output/{0}/gan_disc_model.pt'.format(args.outf+"/models"), 'wb') as f:
      torch.save(gan_disc, f)


def evaluate_autoencoder(data_source, epoch):
    # Turn on evaluation mode which disables dropout.
    autoencoder.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary.word2idx)
    all_accuracies = 0
    bcnt = 0
    for i, batch in enumerate(data_source):
        source, target, lengths = batch
        source = to_gpu(args.cuda, Variable(source, volatile=True))
        target = to_gpu(args.cuda, Variable(target, volatile=True))

        mask = target.gt(0)
        masked_target = target.masked_select(mask)
        # examples x ntokens
        output_mask = mask.unsqueeze(1).expand(mask.size(0), ntokens)

        # output: batch x seq_len x ntokens
        output = autoencoder(source, lengths, noise=True)
        flattened_output = output.view(-1, ntokens)

        masked_output = \
            flattened_output.masked_select(output_mask).view(-1, ntokens)
        total_loss += criterion_ce(masked_output/args.temp, masked_target).data

        # accuracy
        max_vals, max_indices = torch.max(masked_output, 1)
        all_accuracies += \
            torch.mean(max_indices.eq(masked_target).float()).data[0]
        bcnt += 1

        aeoutf = os.environ["DATA_PATH"]+"/arae/output/%s/%d_autoencoder.txt" % (args.outf, epoch)
        with open(aeoutf, "a") as f:
            max_values, max_indices = torch.max(output, 2)
            max_indices = \
                max_indices.view(output.size(0), -1).data.cpu().numpy()
            target = target.view(output.size(0), -1).data.cpu().numpy()
            for t, idx in zip(target, max_indices):
                # real sentence
                chars = " ".join([corpus.dictionary.idx2word[x] for x in t])
                f.write(chars)
                f.write("\n")
                # autoencoder output sentence
                chars = " ".join([corpus.dictionary.idx2word[x] for x in idx])
                f.write(chars)
                f.write("\n\n")

    return total_loss[0] / len(data_source), all_accuracies/bcnt


def evaluate_generator(noise, epoch):
    gan_gen.eval()
    autoencoder.eval()

    # generate from fixed random noise
    fake_hidden = gan_gen(noise)
    max_indices = \
        autoencoder.generate(fake_hidden, args.maxlen, sample=args.sample)

    with open(os.environ["DATA_PATH"]+"/arae/output/%s/%s_generated.txt" % (args.outf, epoch), "w") as f:
        max_indices = max_indices.data.cpu().numpy()
        for idx in max_indices:
            # generated sentence
            words = [corpus.dictionary.idx2word[x] for x in idx]
            # truncate sentences to first occurrence of <eos>
            truncated_sent = []
            for w in words:
                if w != '<eos>':
                    truncated_sent.append(w)
                else:
                    break
            chars = " ".join(truncated_sent)
            f.write(chars)
            f.write("\n")


def evaluate_inverter(data_source, epoch):
    # Turn on evaluation mode which disables dropout.
    autoencoder.eval()
    inverter.eval()
    gan_gen.eval()
    gan_disc.eval()

    for batch in data_source:
        source, target, lengths = batch
        source = to_gpu(args.cuda, Variable(source, volatile=True))
        target = to_gpu(args.cuda, Variable(target, volatile=True))

        # sentence -> encoder -> decoder
        hidden = autoencoder.encode(source, lengths, noise=True)
        ae_indices = autoencoder.generate(hidden, args.maxlen, args.sample)

        # sentence -> encoder -> inverter -> generator -> decoder
        inv_z = inverter(hidden)
        inv_hidden = gan_gen(inv_z)
        eigd_indices = autoencoder.generate(inv_hidden, args.maxlen, args.sample)

        with open(os.environ["DATA_PATH"]+"/arae/output/%s/%s_inverter.txt" % (args.outf, epoch), "a") as f:
            target = target.view(ae_indices.size(0), -1).data.cpu().numpy()
            ae_indices = ae_indices.data.cpu().numpy()
            eigd_indices = eigd_indices.data.cpu().numpy()
            for t, ae, eigd in zip(target, ae_indices, eigd_indices):
                # real sentence
                f.write("# # # original sentence # # #\n")
                chars = " ".join([corpus.dictionary.idx2word[x] for x in t])
                f.write(chars)
                # autoencoder output sentence
                f.write("\n# # # sentence -> encoder -> decoder # # #\n")
                chars = " ".join([corpus.dictionary.idx2word[x] for x in ae])
                f.write(chars)
                # corresponding GAN sentence
                f.write("\n# # # sentence -> encoder -> inverter -> generator "
                        "-> decoder # # #\n")
                chars = " ".join([corpus.dictionary.idx2word[x] for x in eigd])
                f.write(chars)
                f.write("\n\n")


def perturb(data_source, epoch, corpus_test, hybrid=False):
    # Turn on evaluation mode which disables dropout.
    autoencoder.eval()
    
    with open(os.environ["DATA_PATH"]+"/arae/output/%s/%s_perturbation.txt" % (args.outf, epoch),
                  "a") as f:
        for batch in data_source:
            premise, hypothesis, target, premise_words , hypothesise_words, lengths = batch
            premise = to_gpu(args.cuda, premise)
            hypothesis = to_gpu(args.cuda, hypothesis)
            
            c = autoencoder.encode(hypothesis, lengths, noise=False)
            z = inverter(c).data.cpu()
            batch_size = premise.size(0)
            for i in range(batch_size):
                f.write("========================================================\n")
                f.write(" ".join(hypothesise_words[i])+"\n")
                if hybrid:
                    x_adv1, x_adv2, d_adv1, d_adv2, all_adv = search(gan_gen, pred_fn, \
                                (premise[i].unsqueeze(0), hypothesis[i].unsqueeze(0)), \
                                target[i], z[i].view(1,100))
                else:
                    x_adv1, x_adv2, d_adv1, d_adv2, all_adv = search_fast(gan_gen, pred_fn, \
                                (premise[i].unsqueeze(0), hypothesis[i].unsqueeze(0)), \
                                target[i], z[i].view(1,100))
                
                try:
                    hyp_sample_idx1 = autoencoder.generate(x_adv1, 10, True).data.cpu().numpy()[0]
                    hyp_sample_idx2 = autoencoder.generate(x_adv2, 10, True).data.cpu().numpy()[0]
                    words1 = [corpus_test.dictionary.idx2word[x] for x in hyp_sample_idx1]
                    words2 = [corpus_test.dictionary.idx2word[x] for x in hyp_sample_idx2]
                    if "<eos>" in words1:
                        words1 = words1[:words1.index("<eos>")]
                    if "<eos>" in words2:
                        words2 = words2[:words2.index("<eos>")]
                        
                    f.write("\n====================Adversary==========================\n")
                    f.write("Classfier 1 => "+" ".join(words1)+"\t"+str(d_adv1)+"\n")
                    f.write("Classfier 2 => "+" ".join(words2)+"\t"+str(d_adv2)+"\n")
                    f.write("========================================================\n")
                    f.write("\n".join(all_adv)+"\n")
                    f.flush()
                except Exception, e:
                    print(e)
                    print(premise_words)
                    print(hypothesise_words)
                    print("no adversary found for : \n {0} \n {1}\n\n".\
                          format(" ".join(premise_words[i]), " ".join(hypothesise_words[i])))
                        

def compute_ppl(file_path):
    sentences = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    sentences = [l.replace('\n', '') for l in lines]
    lm = load_ngram_lm(os.environ["ROOT"]+"/kenlm/models/snli_3gram.arpa")
    ppl = get_ppl(lm, sentences)
    return ppl


def train_lm(eval_path, save_path):
    # generate examples
    indices = []
    noise = to_gpu(args.cuda, Variable(torch.ones(100, args.z_size)))
    for i in range(1000):
        noise.data.normal_(0, 1)

        fake_hidden = gan_gen(noise)
        max_indices = autoencoder.generate(fake_hidden, args.maxlen)
        indices.append(max_indices.data.cpu().numpy())

    indices = np.concatenate(indices, axis=0)

    # write generated sentences to text file
    with open(save_path+".txt", "w") as f:
        # laplacian smoothing
        for word in corpus.dictionary.word2idx.keys():
            f.write(word+"\n")
        for idx in indices:
            # generated sentence
            words = [corpus.dictionary.idx2word[x] for x in idx]
            # truncate sentences to first occurrence of <eos>
            truncated_sent = []
            for w in words:
                if w != '<eos>':
                    truncated_sent.append(w)
                else:
                    break
            chars = " ".join(truncated_sent)
            f.write(chars+"\n")

    # train language model on generated examples
    lm = train_ngram_lm(kenlm_path=args.kenlm_path,
                        data_path=save_path+".txt",
                        output_path=save_path+".arpa",
                        N=args.N)

    # load sentences to evaluate on
    with open(eval_path, 'r') as f:
        lines = f.readlines()
    sentences = [l.replace('\n', '') for l in lines]
    ppl = get_ppl(lm, sentences)

    return ppl


def train_ae(batch, total_loss_ae, start_time, i):
    autoencoder.train()
    autoencoder.zero_grad()

    source, target, lengths = batch
    source = to_gpu(args.cuda, Variable(source))
    target = to_gpu(args.cuda, Variable(target))

    # Create sentence length mask over padding
    mask = target.gt(0)
    masked_target = target.masked_select(mask)
    # examples x ntokens
    output_mask = mask.unsqueeze(1).expand(mask.size(0), ntokens)


    # output: batch x seq_len x ntokens
    output = autoencoder(source, lengths, noise=True)

    # output_size: batch_size, maxlen, self.ntokens
    flattened_output = output.view(-1, ntokens)

    masked_output = \
        flattened_output.masked_select(output_mask).view(-1, ntokens)
    loss = criterion_ce(masked_output/args.temp, masked_target)
    loss.backward()

    # `clip_grad_norm` to prevent exploding gradient in RNNs / LSTMs
    torch.nn.utils.clip_grad_norm(autoencoder.parameters(), args.clip)
    optimizer_ae.step()

    total_loss_ae += loss.data

    accuracy = None
    if i % args.log_interval == 0 and i > 0:
        # accuracy
        probs = F.softmax(masked_output)
        max_vals, max_indices = torch.max(probs, 1)
        accuracy = torch.mean(max_indices.eq(masked_target).float()).data[0]

        cur_loss = total_loss_ae[0] / args.log_interval
        elapsed = time.time() - start_time
        print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
              'loss {:5.2f} | ppl {:8.2f} | acc {:8.2f}'
              .format(epoch, i, len(train_data),
                      elapsed * 1000 / args.log_interval,
                      cur_loss, math.exp(cur_loss), accuracy))

        with open(os.environ["DATA_PATH"]+"/arae/output/{}/logs.txt".format(args.outf), 'a') as f:
            f.write('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f} | acc {:8.2f}\n'.
                    format(epoch, i, len(train_data),
                           elapsed * 1000 / args.log_interval,
                           cur_loss, math.exp(cur_loss), accuracy))

        total_loss_ae = 0
        start_time = time.time()

    return total_loss_ae, start_time


def train_inv(data_batch=None, gamma=0.5, lengths=None):
    inverter.train()
    inverter.zero_grad()

    noise = to_gpu(args.cuda,
                   Variable(torch.ones(args.batch_size, args.z_size)))
    noise.data.normal_(0, 1)

    if args.use_inv_ae:
        autoencoder.train()
        autoencoder.zero_grad()
        source, target, lengths = data_batch
        source = to_gpu(args.cuda, Variable(source))
        target = to_gpu(args.cuda, Variable(target))
        # Create sentence length mask over padding
        mask = target.gt(0)
        masked_target = target.masked_select(mask)
        # examples x ntokens
        output_mask = mask.unsqueeze(1).expand(mask.size(0), ntokens)
        output = autoencoder(source, lengths, noise=True, generator=gan_gen, inverter=inverter)
        # output_size: batch_size, maxlen, self.ntokens
        flattened_output = output.view(-1, ntokens)

        masked_output = \
            flattened_output.masked_select(output_mask).view(-1, ntokens)
        errI = criterion_ce(masked_output/args.temp, masked_target)

    elif args.useJS:
        fake_hidden = gan_gen(noise)
        inv_noise = inverter(fake_hidden)
        errI = criterion_js(inv_noise, noise)
        if data_batch:
            source, target, lengths = data_batch
            source = to_gpu(args.cuda, Variable(source))
            real_hidden = autoencoder.encode(source, lengths, noise=True)
            real_hidden = to_gpu(args.cuda, Variable(real_hidden.data))
            real_noise = inverter(real_hidden)
            hidden = gan_gen(real_noise)
            errI = gamma * errI
            errI += (1-gamma) * criterion_mse(hidden, real_hidden)
    else:
        fake_hidden = gan_gen(noise)
        inv_noise = inverter(fake_hidden)
        errI = criterion_mse(inv_noise, noise)
        
     # `clip_grad_norm` to prevent exploding gradient in RNNs / LSTMs
    torch.nn.utils.clip_grad_norm(autoencoder.parameters(), args.clip)

    # loss / backprop
    errI.backward()
    
    optimizer_inv.step()

    return errI


def train_gan_g():
    gan_gen.train()
    gan_gen.zero_grad()

    noise = to_gpu(args.cuda,
                   Variable(torch.ones(args.batch_size, args.z_size)))
    noise.data.normal_(0, 1)

    fake_hidden = gan_gen(noise)
    errG = gan_disc(fake_hidden)

    # loss / backprop
    errG.backward(one)
    optimizer_gan_g.step()

    return errG


def train_gan_d(batch):
    # clamp parameters to a cube
    for p in gan_disc.parameters():
        p.data.clamp_(-args.gan_clamp, args.gan_clamp)

    autoencoder.train()
    autoencoder.zero_grad()
    gan_disc.train()
    gan_disc.zero_grad()

    # positive samples ----------------------------
    # generate real codes
    source, target, lengths = batch
    source = to_gpu(args.cuda, Variable(source))
    target = to_gpu(args.cuda, Variable(target))

    # batch_size x nhidden
    real_hidden = autoencoder(source, lengths, noise=False, encode_only=True)
    real_hidden.register_hook(grad_hook)

    # loss / backprop
    errD_real = gan_disc(real_hidden)
    errD_real.backward(one)

    # negative samples ----------------------------
    # generate fake codes
    noise = to_gpu(args.cuda,
                   Variable(torch.ones(args.batch_size, args.z_size)))
    noise.data.normal_(0, 1)

    # loss / backprop
    fake_hidden = gan_gen(noise)
    errD_fake = gan_disc(fake_hidden.detach())
    errD_fake.backward(mone)

    # `clip_grad_norm` to prvent exploding gradient problem in RNNs / LSTMs
    torch.nn.utils.clip_grad_norm(autoencoder.parameters(), args.clip)

    optimizer_gan_d.step()
    optimizer_ae.step()
    errD = -(errD_real - errD_fake)

    return errD, errD_real, errD_fake


def grad_hook(grad):
  # Gradient norm: regularize to be same
  # code_grad_gan * code_grad_ae / norm(code_grad_gan)
  if args.enc_grad_norm:
    gan_norm = torch.norm(grad, 2, 1).detach().data.mean()
    normed_grad = grad * autoencoder.grad_norm / gan_norm
  else:
    normed_grad = grad

  # weight factor and sign flip
  normed_grad *= -math.fabs(args.gan_toenc)
  return normed_grad


print("Training...")
with open(os.environ["DATA_PATH"]+"/arae/output/{}/logs.txt".format(args.outf), 'a') as f:
    f.write('Training...\n')

# schedule of increasing GAN training loops
if args.niters_gan_schedule != "":
    gan_schedule = [int(x) for x in args.niters_gan_schedule.split("-")]
else:
    gan_schedule = []
niter_gan = 1

fixed_noise = to_gpu(args.cuda,
                     Variable(torch.ones(args.batch_size, args.z_size)))
fixed_noise.data.normal_(0, 1)
one = to_gpu(args.cuda, torch.FloatTensor([1]))
mone = one * -1

best_ppl = None
impatience = 0
all_ppl = []
for epoch in range(start_epoch, args.epochs+1):
    # update gan training schedule
    if epoch in gan_schedule:
        niter_gan += 1
        print("GAN training loop schedule increased to {}".format(niter_gan))
        with open(os.environ["DATA_PATH"]+"/arae/output/{}/logs.txt".format(args.outf), 'a') as f:
            f.write("GAN training loop schedule increased to {}\n".
                    format(niter_gan))

    total_loss_ae = 0
    epoch_start_time = time.time()
    start_time = time.time()
    niter = 0
    niter_global = 1
    # loop through all batches in training data
    while niter < len(train_data):

        # train autoencoder ----------------------------
        for i in range(args.niters_ae):
            if args.update_base:
                if niter == len(train_data):
                    break  # end of epoch
                total_loss_ae, start_time = \
                    train_ae(train_data[niter], total_loss_ae, start_time, niter)
            niter += 1

        # train gan ----------------------------------
        for k in range(niter_gan):

            if args.update_base:
                # train discriminator/critic
                for i in range(args.niters_gan_d):
                    # feed a seen sample within this epoch; good for early training
                    errD, errD_real, errD_fake = \
                         train_gan_d(train_data[niter-1])

                # train generator
                for i in range(args.niters_gan_g):
                    errG = train_gan_g()

            # train inverter
            for i in range(args.niters_inv):
                errI = train_inv(train_data[niter-1])

        niter_global += 1
        if args.update_base:

            if niter_global % 100 == 0:
                print('[%d/%d][%d/%d] Loss_D: %.8f (Loss_D_real: %.8f '
                      'Loss_D_fake: %.8f) Loss_G: %.8f Loss_I: %.8f'
                      % (epoch, args.epochs, niter, len(train_data),
                         errD.data[0], errD_real.data[0],
                         errD_fake.data[0], errG.data[0], errI.data[0]))
                with open(os.environ["DATA_PATH"]+"/arae/output/{}/logs.txt".format(args.outf), 'a') as f:
                    f.write('[%d/%d][%d/%d] Loss_D: %.8f (Loss_D_real: %.8f '
                            'Loss_D_fake: %.8f) Loss_G: %.8f Loss_I: %.8f\n'
                            % (epoch, args.epochs, niter, len(train_data),
                               errD.data[0], errD_real.data[0],
                               errD_fake.data[0], errG.data[0], errI.data[0]))

                # exponentially decaying noise on autoencoder
                autoencoder.noise_radius = \
                    autoencoder.noise_radius*args.noise_anneal

                if niter_global % 30000 == 0:
                    evaluate_generator(fixed_noise, "epoch{}_step{}".
                                       format(epoch, niter_global))

                    # evaluate with lm
                    if not args.no_earlystopping and epoch > args.min_epochs:
                        ppl = train_lm(eval_path=os.path.join(args.data_path,
                                                              "test.txt"),
                                       save_path=os.environ["DATA_PATH"]+"/arae/output/{}/"
                                                 "epoch{}_step{}_lm_generations".
                                                 format(args.outf, epoch,
                                                        niter_global))
                        all_ppl.append(ppl)
                        if best_ppl is None or ppl < best_ppl:
                            impatience = 0
                            best_ppl = ppl
                            print("New best ppl {}\n".format(best_ppl))
                            with open(os.environ["DATA_PATH"]+"/arae/output/{}/logs.txt".
                                      format(args.outf), 'a') as f:
                                f.write("New best ppl {}\n".format(best_ppl))
                            save_model()
                        else:
                            impatience += 1
                            # end training
                            if impatience > args.patience:
                                print("Ending training")
                                with open(os.environ["DATA_PATH"]+"/arae/output/{}/logs.txt".
                                          format(args.outf), 'a') as f:
                                    f.write("\nEnding Training\n")
                                sys.exit()

        else:
            if niter_global % 100==0:
                with open(os.environ["DATA_PATH"]+"/arae/output/{}/logs.txt".format(args.outf), 'a') as f:
                    f.write('[%d/%d][%d/%d] Loss_I: %.8f \n'
                          % (epoch, args.epochs, niter, len(train_data), errI.data[0]))

    # end of epoch ----------------------------
    # evaluation
    
    if not args.update_base:
        perturb(test_data, epoch, corpus_test, hybrid=args.hybrid)
        test_data = iter(testloader)
    
    save_model(epoch)

    if ((not args.load_pretrained) and (not args.reload_exp)):
        test_loss, accuracy = evaluate_autoencoder(test_data, epoch)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | '
              'test ppl {:5.2f} | acc {:3.3f}'.
              format(epoch, (time.time() - epoch_start_time),
                     test_loss, math.exp(test_loss), accuracy))
        print('-' * 89)

        with open(os.environ["DATA_PATH"]+"/arae/output/{}/logs.txt".format(args.outf), 'a') as f:
            f.write('-' * 89)
            f.write('\n| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} |'
                    ' test ppl {:5.2f} | acc {:3.3f}\n'.
                    format(epoch, (time.time() - epoch_start_time),
                           test_loss, math.exp(test_loss), accuracy))
            f.write('-' * 89)
            f.write('\n')

        evaluate_generator(fixed_noise, "end_of_epoch_{}".format(epoch))
        if not args.no_earlystopping and epoch >= args.min_epochs:
            ppl = train_lm(eval_path=os.path.join(args.data_path, "test.txt"),
                           save_path=os.environ["DATA_PATH"]+"/arae/output/{}/end_of_epoch{}_lm_generations".
                                     format(args.outf, epoch))
            
            all_ppl.append(ppl)
           
            if best_ppl is None or ppl < best_ppl:
                impatience = 0
                best_ppl = ppl
                print("New best ppl {}\n".format(best_ppl))
                with open(os.environ["DATA_PATH"]+"/arae/output/{}/logs.txt".format(args.outf), 'a') as f:
                    f.write("New best ppl {}\n".format(best_ppl))
                save_model()
            else:
                impatience += 1
                # end training
                if impatience > args.patience:
                    print("Ending training")
                    with open(os.environ["DATA_PATH"]+"/arae/output/{}/logs.txt".format(args.outf), 'a') as f:
                        f.write("\nEnding Training\n")
                    sys.exit()

    # shuffle between epochs
    train_data = batchify(corpus.train, args.batch_size, args.maxlen, packed_rep=args.packed_rep, shuffle=True)
