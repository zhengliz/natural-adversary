import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import to_gpu, load_embeddings
import json
import os
import numpy as np


class MLP_D(nn.Module):
    def __init__(self, ninput, noutput, layers,
                 activation=nn.LeakyReLU(0.2), gpu=False):
        super(MLP_D, self).__init__()
        self.ninput = ninput
        self.noutput = noutput

        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.layers = []

        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
            self.add_module("layer"+str(i+1), layer)

            # No batch normalization after first layer
            if i != 0:
                bn = nn.BatchNorm1d(layer_sizes[i+1], eps=1e-05, momentum=0.1)
                self.layers.append(bn)
                self.add_module("bn"+str(i+1), bn)

            self.layers.append(activation)
            self.add_module("activation"+str(i+1), activation)

        layer = nn.Linear(layer_sizes[-1], noutput)
        self.layers.append(layer)
        self.add_module("layer"+str(len(self.layers)), layer)

        self.init_weights()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = torch.mean(x)
        return x

    def init_weights(self):
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass


class MLP_G(nn.Module):
    def __init__(self, ninput, noutput, layers,
                 activation=nn.ReLU(), gpu=False):
        super(MLP_G, self).__init__()
        self.ninput = ninput
        self.noutput = noutput
        self.gpu = gpu

        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.layers = []

        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
            self.add_module("layer"+str(i+1), layer)

            bn = nn.BatchNorm1d(layer_sizes[i+1], eps=1e-05, momentum=0.1)
            self.layers.append(bn)
            self.add_module("bn"+str(i+1), bn)

            self.layers.append(activation)
            self.add_module("activation"+str(i+1), activation)

        layer = nn.Linear(layer_sizes[-1], noutput)
        self.layers.append(layer)
        self.add_module("layer"+str(len(self.layers)), layer)

        self.init_weights()

    def forward(self, x):
        if x.__class__.__name__ =="ndarray":
            x = Variable(torch.FloatTensor(x)).cuda()
            #x = x.cpu()
        if x.__class__.__name__ =="FloatTensor":
            x = Variable(x).cuda()
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

    def init_weights(self):
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass


class MLP_I(nn.Module):
    # separate Inverter to map continuous code back to z
    def __init__(self, ninput, noutput, layers,
                 activation=nn.ReLU(), gpu=False):
        super(MLP_I, self).__init__()
        self.ninput = ninput
        self.noutput = noutput

        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.layers = []

        for i in range(len(layer_sizes) - 1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            self.layers.append(layer)
            self.add_module("layer" + str(i + 1), layer)

            bn = nn.BatchNorm1d(layer_sizes[i + 1], eps=1e-05, momentum=0.1)
            self.layers.append(bn)
            self.add_module("bn" + str(i + 1), bn)

            self.layers.append(activation)
            self.add_module("activation" + str(i + 1), activation)

        layer = nn.Linear(layer_sizes[-1], noutput)
        self.layers.append(layer)
        self.add_module("layer" + str(len(self.layers)), layer)

        self.init_weights()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

    def init_weights(self):
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass


class MLP_I_AE(nn.Module):
    # separate Inverter to map continuous code back to z (mean & std)
    def __init__(self, ninput, noutput, layers,
                 activation=nn.ReLU(), gpu=False):
        super(MLP_I_AE, self).__init__()
        self.ninput = ninput
        self.noutput = noutput
        self.gpu = gpu
        noutput_mu = noutput
        noutput_var = noutput
        
        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
        self.layers = []

        for i in range(len(layer_sizes) - 1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            self.layers.append(layer)
            self.add_module("layer" + str(i + 1), layer)

            bn = nn.BatchNorm1d(layer_sizes[i + 1], eps=1e-05, momentum=0.1)
            self.layers.append(bn)
            self.add_module("bn" + str(i + 1), bn)

            self.layers.append(activation)
            self.add_module("activation" + str(i + 1), activation)

        layer = nn.Linear(layer_sizes[-1], noutput)
        self.layers.append(layer)
        self.add_module("layer" + str(len(self.layers)), layer)

        self.linear_mu = nn.Linear(noutput, noutput_mu)
        self.linear_var = nn.Linear(noutput, noutput_var)
        
        self.init_weights()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        mu = self.linear_mu(x)
        logvar = self.linear_var(x)    
        std = 0.5*logvar
        std = std.exp_()                                        # std
        epsilon = Variable(std.data.new(std.size()).normal_())  # normal noise with the same type and size as std.data
        if self.gpu:
            epsilon = epsilon.cuda()
        
        sample = mu + (epsilon * std)
        
        return sample

    def init_weights(self):
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass
            
        self.linear_mu.weight.data.normal_(0, init_std)
        self.linear_mu.bias.data.fill_(0)
        self.linear_var.weight.data.normal_(0, init_std)
        self.linear_var.bias.data.fill_(0)


class Seq2SeqCAE(nn.Module):
    # CNN encoder, LSTM decoder
    def __init__(self, emsize, nhidden, ntokens, nlayers, conv_windows="5-5-3", conv_strides="2-2-2",
                 conv_layer="500-700-1000", activation=nn.LeakyReLU(0.2, inplace=True),
                 noise_radius=0.2, hidden_init=False, dropout=0, gpu=True):
        super(Seq2SeqCAE, self).__init__()
        self.nhidden = nhidden      # size of hidden vector in LSTM
        self.emsize = emsize
        self.ntokens = ntokens
        self.nlayers = nlayers
        self.noise_radius = noise_radius
        self.hidden_init = hidden_init
        self.dropout = dropout
        self.gpu = gpu
        self.arch_conv_filters = conv_layer
        self.arch_conv_strides = conv_strides
        self.arch_conv_windows = conv_windows
        self.start_symbols = to_gpu(gpu, Variable(torch.ones(10, 1).long()))

        # Vocabulary embedding
        self.embedding = nn.Embedding(ntokens, emsize)
        self.embedding_decoder = nn.Embedding(ntokens, emsize)

        conv_layer_sizes = [emsize] + [int(x) for x in conv_layer.split('-')]
        conv_strides_sizes = [int(x) for x in conv_strides.split('-')]
        conv_windows_sizes = [int(x) for x in conv_windows.split('-')]
        self.encoder = nn.Sequential()

        for i in range(len(conv_layer_sizes) - 1):
            layer = nn.Conv1d(conv_layer_sizes[i], conv_layer_sizes[i + 1], \
                              conv_windows_sizes[i], stride=conv_strides_sizes[i])
            self.encoder.add_module("layer-" + str(i + 1), layer)

            bn = nn.BatchNorm1d(conv_layer_sizes[i + 1])
            self.encoder.add_module("bn-" + str(i + 1), bn)

            self.encoder.add_module("activation-" + str(i + 1), activation)

        self.linear = nn.Linear(1000, emsize)

        decoder_input_size = emsize + nhidden
        self.decoder = nn.LSTM(input_size=decoder_input_size,
                               hidden_size=nhidden,
                               num_layers=1,
                               dropout=dropout,
                               batch_first=True)
        self.linear_dec = nn.Linear(nhidden, ntokens)

        # 9-> 7-> 3 -> 1
    def decode(self, hidden, batch_size, maxlen, indices=None, lengths=None):
        # batch x hidden
        all_hidden = hidden.unsqueeze(1).repeat(1, maxlen, 1)

        if self.hidden_init:
            # initialize decoder hidden state to encoder output
            state = (hidden.unsqueeze(0), self.init_state(batch_size))
        else:
            state = self.init_hidden(batch_size)

        embeddings = self.embedding_decoder(indices)        # training stage
        augmented_embeddings = torch.cat([embeddings, all_hidden], 2)

        output, state = self.decoder(augmented_embeddings, state)

        decoded = self.linear_dec(output.contiguous().view(-1, self.nhidden))
        decoded = decoded.view(batch_size, maxlen, self.ntokens)

        return decoded

    def generate(self, hidden, maxlen, sample=True, temp=1.0):
        """Generate through decoder; no backprop"""
        if hidden.ndimension() == 1:
            hidden = hidden.unsqueeze(0)
        batch_size = hidden.size(0)

        if self.hidden_init:
            # initialize decoder hidden state to encoder output
            state = (hidden.unsqueeze(0), self.init_state(batch_size))
        else:
            state = self.init_hidden(batch_size)

        if not self.gpu:
            self.start_symbols = self.start_symbols.cpu()
        # <sos>
        self.start_symbols.data.resize_(batch_size, 1)
        self.start_symbols.data.fill_(1)

        embedding = self.embedding_decoder(self.start_symbols)
        inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)

        # unroll
        all_indices = []
        for i in range(maxlen):
            output, state = self.decoder(inputs, state)
            overvocab = self.linear_dec(output.squeeze(1))

            if not sample:
                vals, indices = torch.max(overvocab, 1)
            else:
                # sampling
                probs = F.softmax(overvocab/temp)
                indices = torch.multinomial(probs, 1)

            if indices.ndimension()==1:
                indices = indices.unsqueeze(1)
            all_indices.append(indices)

            embedding = self.embedding_decoder(indices)
            inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)

        max_indices = torch.cat(all_indices, 1)

        return max_indices


    def init_weights(self):
        initrange = 0.1

        # Initialize Vocabulary Matrix Weight
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.embedding.weight.data[0].zero()
        self.embedding_decoder.weight.data.uniform_(-initrange, initrange)

        # Initialize Encoder and Decoder Weights
        for p in self.encoder.parameters():
            p.data.uniform_(-initrange, initrange)
        for p in self.decoder.parameters():
            p.data.uniform_(-initrange, initrange)

        # Initialize Linear Weight
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.fill_(0)

    def encode(self, indices, lengths, noise):
        embeddings = self.embedding(indices)
        embeddings = embeddings.transpose(1,2)
        c_pre_lin = self.encoder(embeddings)
        c_pre_lin = c_pre_lin.squeeze(2)
        hidden = self.linear(c_pre_lin)
        # normalize to unit ball (l2 norm of 1) - p=2, dim=1
        norms = torch.norm(hidden, 2, 1)
        if norms.ndimension()==1:
            norms=norms.unsqueeze(1)
        hidden = torch.div(hidden, norms.expand_as(hidden))

        if noise and self.noise_radius > 0:
            gauss_noise = torch.normal(means=torch.zeros(hidden.size()),
                                       std=self.noise_radius)
            if self.gpu:
                gauss_noise = gauss_noise.cuda()
                
            hidden = hidden + to_gpu(self.gpu, Variable(gauss_noise))

        return hidden

    def init_hidden(self, bsz):
        zeros1 = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        zeros2 = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        return to_gpu(self.gpu, zeros1), to_gpu(self.gpu, zeros2) # (hidden, cell)

    def init_state(self, bsz):
        zeros = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        return to_gpu(self.gpu, zeros)

    def store_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        return grad

    def forward(self, indices, lengths, noise, encode_only=False, generator=None, inverter=None):
        if not generator:   # only enc -> dec
            batch_size, maxlen = indices.size()
            self.embedding.weight.data[0].fill_(0)
            self.embedding_decoder.weight.data[0].fill_(0)
            hidden = self.encode(indices, lengths, noise)
            if encode_only:
                return hidden

            if hidden.requires_grad:
                hidden.register_hook(self.store_grad_norm)

            decoded = self.decode(hidden, batch_size, maxlen,
                              indices=indices, lengths=lengths)
        else:               # enc -> inv -> gen -> dec
            batch_size, maxlen = indices.size()
            self.embedding.weight.data[0].fill_(0)
            self.embedding_decoder.weight.data[0].fill_(0)
            hidden = self.encode(indices, lengths, noise)
            if encode_only:
                return hidden

            if hidden.requires_grad:
                hidden.register_hook(self.store_grad_norm)

            z_hat = inverter(hidden)
            c_hat = generator(z_hat)

            decoded = self.decode(c_hat, batch_size, maxlen,
                              indices=indices, lengths=lengths)

        return decoded


class Seq2Seq(nn.Module):
    def __init__(self, emsize, nhidden, ntokens, nlayers, noise_radius=0.2,
                 hidden_init=False, dropout=0, gpu=True):
        super(Seq2Seq, self).__init__()
        self.nhidden = nhidden
        self.emsize = emsize
        self.ntokens = ntokens
        self.nlayers = nlayers
        self.noise_radius = noise_radius
        self.hidden_init = hidden_init
        self.dropout = dropout
        self.gpu = gpu

        self.start_symbols = to_gpu(gpu, Variable(torch.ones(10, 1).long()))

        # Vocabulary embedding
        self.embedding = nn.Embedding(ntokens, emsize)
        self.embedding_decoder = nn.Embedding(ntokens, emsize)

        # RNN Encoder and Decoder
        self.encoder = nn.LSTM(input_size=emsize,
                               hidden_size=nhidden,
                               num_layers=nlayers,
                               dropout=dropout,
                               batch_first=True)

        decoder_input_size = emsize+nhidden
        self.decoder = nn.LSTM(input_size=decoder_input_size,
                               hidden_size=nhidden,
                               num_layers=1,
                               dropout=dropout,
                               batch_first=True)

        # Initialize Linear Transformation
        self.linear = nn.Linear(nhidden, ntokens)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1

        # Initialize Vocabulary Matrix Weight
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.embedding_decoder.weight.data.uniform_(-initrange, initrange)

        # Initialize Encoder and Decoder Weights
        for p in self.encoder.parameters():
            p.data.uniform_(-initrange, initrange)
        for p in self.decoder.parameters():
            p.data.uniform_(-initrange, initrange)

        # Initialize Linear Weight
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.fill_(0)

    def init_hidden(self, bsz):
        zeros1 = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        zeros2 = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        return (to_gpu(self.gpu, zeros1), to_gpu(self.gpu, zeros2)) # (hidden, cell)

    def init_state(self, bsz):
        zeros = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        return to_gpu(self.gpu, zeros)

    def store_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        return grad

    def forward(self, indices, lengths, noise, encode_only=False, generator=None, inverter=None):
        if not generator:
            batch_size, maxlen = indices.size()

            hidden = self.encode(indices, lengths, noise)

            if encode_only:
                return hidden

            if hidden.requires_grad:
                hidden.register_hook(self.store_grad_norm)

            decoded = self.decode(hidden, batch_size, maxlen,
                                  indices=indices, lengths=lengths)
        else:
            batch_size, maxlen = indices.size()
            self.embedding.weight.data[0].fill_(0)
            self.embedding_decoder.weight.data[0].fill_(0)
            hidden = self.encode(indices, lengths, noise)
            if encode_only:
                return hidden

            if hidden.requires_grad:
                hidden.register_hook(self.store_grad_norm)

            z_hat = inverter(hidden)
            c_hat = generator(z_hat)

            decoded = self.decode(c_hat, batch_size, maxlen,
                              indices=indices, lengths=lengths)

        return decoded

    def encode(self, indices, lengths, noise):
        embeddings = self.embedding(indices)
        packed_embeddings = pack_padded_sequence(input=embeddings,
                                                 lengths=lengths,
                                                 batch_first=True)

        # Encode
        packed_output, state = self.encoder(packed_embeddings)

        hidden, cell = state
        # batch_size x nhidden
        hidden = hidden[-1]  # get hidden state of last layer of encoder

        # normalize to unit ball (l2 norm of 1) - p=2, dim=1
        norms = torch.norm(hidden, 2, 1)
        if norms.ndimension()==1:
            norms=norms.unsqueeze(1)
        hidden = torch.div(hidden, norms.expand_as(hidden))

        if noise and self.noise_radius > 0:
            gauss_noise = torch.normal(means=torch.zeros(hidden.size()),
                                       std=self.noise_radius)
            hidden = hidden + to_gpu(self.gpu, Variable(gauss_noise))

        return hidden

    def decode(self, hidden, batch_size, maxlen, indices=None, lengths=None):
        # batch x hidden
        all_hidden = hidden.unsqueeze(1).repeat(1, maxlen, 1)

        if self.hidden_init:
            # initialize decoder hidden state to encoder output
            state = (hidden.unsqueeze(0), self.init_state(batch_size))
        else:
            state = self.init_hidden(batch_size)

        embeddings = self.embedding_decoder(indices)
        augmented_embeddings = torch.cat([embeddings, all_hidden], 2)
        packed_embeddings = pack_padded_sequence(input=augmented_embeddings,
                                                 lengths=lengths,
                                                 batch_first=True)

        packed_output, state = self.decoder(packed_embeddings, state)
        output, lengths = pad_packed_sequence(packed_output, batch_first=True)
        
        # reshape to batch_size*maxlen x nhidden before linear over vocab
        decoded = self.linear(output.contiguous().view(-1, self.nhidden))
        decoded = decoded.view(batch_size, maxlen, self.ntokens)

        return decoded

    def generate(self, hidden, maxlen, sample=True, temp=1.0):
        """Generate through decoder; no backprop"""

        batch_size = hidden.size(0)

        if self.hidden_init:
            # initialize decoder hidden state to encoder output
            state = (hidden.unsqueeze(0), self.init_state(batch_size))
        else:
            state = self.init_hidden(batch_size)

        # <sos>
        self.start_symbols.data.resize_(batch_size, 1)
        self.start_symbols.data.fill_(1)

        embedding = self.embedding_decoder(self.start_symbols)
        inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)

        # unroll
        all_indices = []
        for i in range(maxlen):
            output, state = self.decoder(inputs, state)
            overvocab = self.linear(output.squeeze(1))

            if not sample:
                vals, indices = torch.max(overvocab, 1)
            else:
                # sampling
                probs = F.softmax(overvocab/temp)
                indices = torch.multinomial(probs, 1)

            if indices.ndimension()==1:
                indices = indices.unsqueeze(1)
            all_indices.append(indices)

            embedding = self.embedding_decoder(indices)
            inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)

        max_indices = torch.cat(all_indices, 1)

        return max_indices


def load_models(load_path):
    model_args = json.load(open("{}/args.json".format(load_path), "r"))
    word2idx = json.load(open("{}/vocab.json".format(load_path), "r"))
    idx2word = {v: k for k, v in word2idx.items()}

    print('Loading models from' + load_path+"/models")
    ae_path = os.path.join(load_path+"/models/", "autoencoder_model.pt")
    inv_path = os.path.join(load_path+"/models/", "inverter_model.pt")
    gen_path = os.path.join(load_path+"/models/", "gan_gen_model.pt")
    disc_path = os.path.join(load_path+"/models/", "gan_disc_model.pt")

    autoencoder = torch.load(ae_path)
    inverter = torch.load(inv_path)
    gan_gen = torch.load(gen_path)
    gan_disc = torch.load(disc_path)
    return model_args, idx2word, autoencoder, inverter, gan_gen, gan_disc


def generate(autoencoder, gan_gen, z, vocab, sample, maxlen):
    """
    Assume noise is batch_size x z_size
    """
    if type(z) == Variable:
        noise = z
    elif type(z) == torch.FloatTensor or type(z) == torch.cuda.FloatTensor:
        noise = Variable(z, volatile=True)
    elif type(z) == np.ndarray:
        noise = Variable(torch.from_numpy(z).float(), volatile=True)
    else:
        raise ValueError("Unsupported input type (noise): {}".format(type(z)))

    gan_gen.eval()
    autoencoder.eval()

    # generate from random noise
    fake_hidden = gan_gen(noise)
    max_indices = autoencoder.generate(hidden=fake_hidden,
                                       maxlen=maxlen,
                                       sample=sample)

    max_indices = max_indices.data.cpu().numpy()
    sentences = []
    for idx in max_indices:
        # generated sentence
        words = [vocab[x] for x in idx]
        # truncate sentences to first occurrence of <eos>
        truncated_sent = []
        for w in words:
            if w != '<eos>':
                truncated_sent.append(w)
            else:
                break
        sent = " ".join(truncated_sent)
        sentences.append(sent)

    return sentences


class JSDistance(nn.Module):
    def __init__(self, mean=0, std=1, epsilon=1e-5):
        super(JSDistance, self).__init__()
        self.epsilon = epsilon
        self.distrib_type_normal = True

    def get_kl_div(self, input, target):
        src_mu = torch.mean(input)
        src_std = torch.std(input)
        tgt_mu = torch.mean(target)
        tgt_std = torch.std(target)
        kl = torch.log(tgt_std/src_std) - 0.5 +\
                    (src_std ** 2 + (src_mu - tgt_mu) ** 2)/(2 * (tgt_std ** 2))
        return kl

    def forward(self, input, target):
        ##KL(p, q) = log(sig2/sig1) + ((sig1^2 + (mu1 - mu2)^2)/2*sig2^2) - 1/2
        if self.distrib_type_normal:
            d1=self.get_kl_div(input, target)
            d2=self.get_kl_div(target, input)
            return 0.5 * (d1+d2)
        else:
            input_num_zero = input.data[torch.eq(input.data, 0)]
            if input_num_zero.dim() > 0:
                input_num_zero = input_num_zero.size(0)
                input.data = input.data - (self.epsilon/input_num_zero)
                input.data[torch.lt(input.data, 0)] = self.epsilon/input_num_zero
            target_num_zero = target.data[torch.eq(target.data, 0)]
            if target_num_zero.dim() > 0:
                target_num_zero = target_num_zero.size(0)
                target.data = target.data - (self.epsilon/target_num_zero)
                target.data[torch.lt(target.data, 0)] = self.epsilon/target_num_zero
            d1 = torch.sum(input * torch.log(input/target))/input.size(0)
            d2 = torch.sum(target * torch.log(target/input))/input.size(0)
            return (d1+d2)/2


class Baseline_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, maxlen=10, dropout= 0, vocab_size=11004, gpu=False):
        super(Baseline_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.nlayers = 1
        self.gpu = gpu
        self.maxlen = maxlen
        self.embedding_prem = nn.Embedding(vocab_size+4, emb_size)
        self.embedding_hypo = nn.Embedding(vocab_size+4, emb_size)
        self.premise_encoder = nn.LSTM(input_size=emb_size,
                               hidden_size=hidden_size,
                               num_layers=1,
                               dropout=dropout,
                               batch_first=True)
        print(self.premise_encoder)
        self.hypothesis_encoder = nn.LSTM(input_size=emb_size,
                               hidden_size=hidden_size,
                               num_layers=1,
                               dropout=dropout,
                               batch_first=True)
        self.layers = nn.Sequential()
        layer_sizes = [2*hidden_size, 400, 100]
        for i in range(len(layer_sizes) - 1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i + 1])
            self.layers.add_module("layer" + str(i + 1), layer)
            
            bn = nn.BatchNorm1d(layer_sizes[i + 1], eps=1e-05, momentum=0.1)
            self.layers.add_module("bn" + str(i + 1), bn)

            self.layers.add_module("activation" + str(i + 1), nn.ReLU())

        layer = nn.Linear(layer_sizes[-1], 3)
        self.layers.add_module("layer" + str(len(layer_sizes)), layer)
        
        self.layers.add_module("softmax", nn.Softmax())
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
    
        # Initialize Vocabulary Matrix Weight
        self.embedding_prem.weight.data.uniform_(-initrange, initrange)
        self.embedding_hypo.weight.data.uniform_(-initrange, initrange)
    
        # Initialize Encoder and Decoder Weights
        for p in self.premise_encoder.parameters():
            p.data.uniform_(-initrange, initrange)
        for p in self.hypothesis_encoder.parameters():
            p.data.uniform_(-initrange, initrange)
    
        # Initialize Linear Weight
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass

    def init_hidden(self, bsz):
        zeros1 = Variable(torch.zeros(self.nlayers, bsz, self.hidden_size))
        zeros2 = Variable(torch.zeros(self.nlayers, bsz, self.hidden_size))
        return (to_gpu(self.gpu, zeros1), to_gpu(self.gpu, zeros2)) # (hidden, cell)

    def store_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        return grad

        
    def forward(self, batch):
        premise_indices, hypothesis_indices = batch
        batch_size = premise_indices.size(0)
        state_prem= self.init_hidden(batch_size)
        state_hypo= self.init_hidden(batch_size)
        premise = self.embedding_prem(premise_indices)
        output_prem, (hidden_prem, _) = self.premise_encoder(premise, state_prem)
        hidden_prem= hidden_prem[-1]
        if hidden_prem.requires_grad:
            hidden_prem.register_hook(self.store_grad_norm)
                
        hypothesis = self.embedding_hypo(hypothesis_indices)
        output_hypo, (hidden_hypo, _) = self.hypothesis_encoder(hypothesis, state_hypo)
        hidden_hypo= hidden_hypo[-1]
        if hidden_hypo.requires_grad:
            hidden_hypo.register_hook(self.store_grad_norm)
            
        concatenated = torch.cat([hidden_prem, hidden_hypo], 1)
        probs = self.layers(concatenated)
        return probs


class Baseline_Embeddings(nn.Module):
    def __init__(self, emb_size, vocab_size=11004):
        super(Baseline_Embeddings, self).__init__()
        self.embedding_prem = nn.Embedding(vocab_size, emb_size)
        self.embedding_hypo = nn.Embedding(vocab_size, emb_size)
        self.linear = nn.Linear(emb_size*2, 3)
        embeddings_mat = load_embeddings()
        self.embedding_prem.weight.data.copy_(embeddings_mat)
        self.embedding_hypo.weight.data.copy_(embeddings_mat)
        
    def forward(self, batch):
        premise_indices, hypothesis_indices = batch
        enc_premise = self.embedding_prem(premise_indices)
        enc_hypothesis = self.embedding_hypo(hypothesis_indices)
        enc_premise = torch.mean(enc_premise,1).squeeze(1)
        enc_hypothesis = torch.mean(enc_hypothesis,1).squeeze(1)
        
        concatenated = torch.cat([enc_premise, enc_hypothesis], 1)
        probs = self.linear(concatenated)
        return probs