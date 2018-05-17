import argparse
import numpy as np
import random
import os

import torch
from torch.autograd import Variable

from utils import Corpus, batchify, to_gpu
from models import load_models


def perturb(data_batch, autoencoder, vocab, sample, maxlen,
            left=0., right=1., n_samples=5, epoch=0, gpu=True):

    autoencoder.eval()

    source, target, lengths = data_batch
    source = to_gpu(gpu, Variable(source, volatile=True))
    target = to_gpu(gpu, Variable(target, volatile=True))

    hidden = autoencoder.encode(source, lengths, noise=True)
    hidden_data = hidden.data
    # n = hidden_data.size()[0]

    c = hidden_data[0].repeat(n_samples, 1)
    delta = to_gpu(gpu, torch.FloatTensor(c.size()).uniform_(left, right))
    c_delta = Variable(c + delta, volatile=True)

    indices = autoencoder.generate(c_delta, maxlen, sample)

    target = target.view(-1).cpu().data.numpy()
    indices = indices.cpu().data.numpy()

    if args.test:
        if not os.path.isdir(args.load_path + '/test'):
            os.makedirs(args.load_path + '/test')
        fin = os.path.realpath(args.load_path) + \
              "/test/input{}.txt".format(epoch)
    else:
        if not os.path.isdir(args.load_path + '/train'):
            os.makedirs(args.load_path + '/train')
        fin = os.path.realpath(args.load_path) + \
              "/train/input{}.txt".format(epoch)

    with open(fin, "w") as f:
        chars = " ".join([vocab[x] for x in target])
        f.write(chars.split(' <eos>')[0].split('.')[0] + " .\n")

        for j in range(n_samples):
            chars = " ".join([vocab[x] for x in indices[j]])
            f.write(chars.split(' <eos>')[0].split('.')[0] + " .\n")

    return check(fin, epoch)


def check(fin, epoch):
    if args.test:
        fout = os.path.realpath(args.load_path) + \
               "/test/pertubation{}.txt".format(epoch)
    else:
        fout = os.path.realpath(args.load_path) + \
               "/train/pertubation{}.txt".format(epoch)

    cmd = "cd /home/zhengliz/Models/sst/ && " + \
          "java -cp \"*\" -mx5g edu.stanford.nlp.sentiment.SentimentPipeline " \
          "-file {} > {}".format(fin, fout)
    os.system(cmd)

    with open(fout, "r") as f:
        content = f.readlines()
    # content = [x.strip() for x in content]

    for row in range(1, len(content) / 2):
        if content[2 * row + 1] != content[1]:
            return True
    return False


def main(args):

    ###########################################################################
    # Load the models
    ###########################################################################

    model_args, idx2word, autoencoder, inverter, gan_gen, gan_disc = \
        load_models(args.load_path)

    # Set the random seed manually for reproducibility.
    random.seed(model_args['seed'])
    np.random.seed(model_args['seed'])
    torch.manual_seed(model_args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(model_args['seed'])
    else:
        print("Note that our pre-trained models require CUDA to evaluate.")

    ###########################################################################
    # Load data
    ###########################################################################

    corpus = Corpus(model_args['data_path'],
                    maxlen=model_args['maxlen'],
                    vocab_size=model_args['vocab_size'],
                    lowercase=model_args['lowercase'])
    if args.test:
        eval_batch_size = 1
        test_data = batchify(corpus.test, eval_batch_size, shuffle=False)
    else:
        train_data = batchify(corpus.train, model_args['batch_size'], shuffle=True)

    print("Loaded data!")

    ###########################################################################
    # Perturbations
    ###########################################################################

    ring_rng = np.linspace(0., 1., 100)
    n_rng = len(test_data) if args.test else len(train_data)

    for idx in range(n_rng):
        data_batch = test_data[idx] if args.test else train_data[idx]

        for l, r in zip(ring_rng, ring_rng[1:]):

            flg = perturb(data_batch, autoencoder, idx2word,
                          model_args['sample'], model_args['maxlen'],
                          left=l, right=r, n_samples=5, epoch=idx,
                          gpu=model_args['cuda'])
            if flg: break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ARAE for Text Eval')
    parser.add_argument('--load_path', type=str, required=True,
                        help='directory to load models from')
    parser.add_argument('--test', action='store_true',
                        help='eval using testing instead of training data')
    args = parser.parse_args()
    print(vars(args))
    main(args)
