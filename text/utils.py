import os
import torch
import numpy as np
import random
import json
import torch.utils.data as data
import pickle as pkl
from torch.autograd import Variable
from copy import deepcopy


def load_kenlm():
    global kenlm
    import kenlm


def to_gpu(gpu, var):
    if gpu:
        return var.cuda()
    return var


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.lvt=True
        self.word2idx['<pad>'] = 0
        self.word2idx['<sos>'] = 1
        self.word2idx['<eos>'] = 2
        self.word2idx['<oov>'] = 3
        self.wordcounts = {}

    # to track word counts
    def add_word(self, word):
        if word not in self.wordcounts:
            self.wordcounts[word] = 1
            if not self.lvt:
                self.word2idx[word] = len(self.word2idx)
        else:
            self.wordcounts[word] += 1

    # prune vocab based on count k cutoff or most frequently seen k words
    def prune_vocab(self, k=5, cnt=False):
        # get all words and their respective counts
        vocab_list = [(word, count) for word, count in self.wordcounts.items()]
        if cnt:
            # prune by count
            self.pruned_vocab = \
                    {pair[0]: pair[1] for pair in vocab_list if pair[1] > k}
        else:
            # prune by most frequently seen words
            vocab_list.sort(key=lambda x: (x[1], x[0]), reverse=True)
            k = min(k, len(vocab_list))
            self.pruned_vocab = [pair[0] for pair in vocab_list[:k]]
        # sort to make vocabulary determistic
        self.pruned_vocab.sort()

        # add all chosen words to new vocabulary/dict
        for word in self.pruned_vocab:
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
        print("original vocab {}; pruned to {}".
              format(len(self.wordcounts), len(self.word2idx)))
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def __len__(self):
        return len(self.word2idx)


class Corpus(object):
    def __init__(self, path, maxlen, vocab_size=11000, lowercase=False, load_vocab=None):
        self.dictionary = Dictionary()
        self.maxlen = maxlen
        self.lowercase = lowercase
        self.vocab_size = vocab_size
        self.train_path = os.path.join(path, 'train.txt')
        self.test_path = os.path.join(path, 'test.txt')

        # make the vocabulary from training set
        if load_vocab:
            self.dictionary.word2idx = json.load(open(load_vocab))
            self.dictionary.idx2word = {v: k for k, v in self.dictionary.word2idx.items()}
            print("Finished loading vocabulary from "+load_vocab+"\n")
        else:
            self.make_vocab()
            
        if os.path.exists(self.train_path):
            self.train = self.tokenize(self.train_path)
        if os.path.exists(self.test_path):
            self.test = self.tokenize(self.test_path)

    def get_indices(self, words):
        vocab = self.dictionary.word2idx
        unk_idx = vocab['<oov>']
        return torch.LongTensor([[vocab[w] if w in vocab else unk_idx for w in words]])
    
    def make_vocab(self):
        assert os.path.exists(self.train_path)
        # Add words to the dictionary
        with open(self.train_path, 'r') as f:
            for line in f:
                if self.lowercase:
                    # -1 to get rid of \n character
                    words = line[:-1].lower().split(" ")
                else:
                    words = line[:-1].split(" ")
                for word in words:
                    word = word.decode('Windows-1252').encode('utf-8')
                    self.dictionary.add_word(word)

        # prune the vocabulary
        self.dictionary.prune_vocab(k=self.vocab_size, cnt=False)

    def tokenize(self, path):
        """Tokenizes a text file."""
        dropped = 0
        
        with open(path, 'r') as f:
            linecount = 0
            lines = []
            for line in f:
                linecount += 1
                if self.lowercase:
                    words = line[:-1].lower().strip().split(" ")
                else:
                    words = line[:-1].strip().split(" ")
                if len(words) > self.maxlen-1:
                    dropped += 1
                    continue
                lens = len(words) + 1
                words = ['<sos>'] + words
                words += ['<eos>']
                
                # vectorize
                vocab = self.dictionary.word2idx
                unk_idx = vocab['<oov>']
                indices = [vocab[w] if w in vocab else unk_idx for w in words]
                #lines.append(indices)
                lines.append((indices, lens))

        print("Number of sentences dropped from {}: {} out of {} total".
              format(path, dropped, linecount))
        return lines


def batchify(data, bsz, max_len, packed_rep=False, shuffle=False, gpu=False):
    if shuffle:
        random.shuffle(data)
    nbatch = len(data) // bsz
    batches = []

    for i in range(nbatch):
        # Pad batches to maximum sequence length in batch
        #batch = data[i*bsz:(i+1)*bsz]
        batch, lengths = zip(*(data[i*bsz:(i+1)*bsz]))
        # subtract 1 from lengths b/c includes BOTH starts & end symbols
        #lengths = [len(x)-1 for x in batch]
        # sort items by length (decreasing)
        batch, lengths = length_sort(batch, lengths)

        # source has no end symbol
        source = [x[:-1] for x in batch]
        # target has no start symbol
        target = [x[1:] for x in batch]
        
        
        # find length to pad to
        if packed_rep:
            maxlen = max(lengths)
        else:
            maxlen = max_len
            
        lengths = [min(x, max_len) for x in lengths]
        #lengths = [max(x, max_len) for x in lengths]
        
        count = 0
        for x, y in zip(source, target):
            if len(x)> maxlen:
                source[count] = x[:maxlen]
            if len(y)>maxlen:
                target[count] = y[:maxlen]
                 
            zeros = (maxlen-len(x))*[0]
            x += zeros
            y += zeros
            count+=1

        
        source = torch.LongTensor(np.array(source))
        target = torch.LongTensor(np.array(target)).view(-1)

        if gpu:
            source = source.cuda()
            target = target.cuda()

        batches.append((source, target, lengths))

    return batches


def length_sort(items, lengths, descending=True):
    """In order to use pytorch variable length sequence package"""
    items = list(zip(items, lengths))
    items.sort(key=lambda x: x[1], reverse=True)
    items, lengths = zip(*items)
    return list(items), list(lengths)


def train_ngram_lm(kenlm_path, data_path, output_path, N):
    """
    Trains a modified Kneser-Ney n-gram KenLM from a text file.
    Creates a .arpa file to store n-grams.
    """
    # create .arpa file of n-grams
    curdir = os.path.abspath(os.path.curdir)
    #
    command = "bin/lmplz -o "+str(N)+" <"+os.path.join(curdir, data_path) + \
              " >"+os.path.join(curdir, output_path)
    os.system("cd "+os.path.join(kenlm_path, 'build')+" && "+command)
    
    load_kenlm()
    # create language model
    model = kenlm.Model(output_path)

    return model


def load_ngram_lm(model_path):
    load_kenlm()
    model = kenlm.Model(model_path)
    return model


def get_ppl(lm, sentences):
    """
    Assume sentences is a list of strings (space delimited sentences)
    """
    total_nll = 0
    total_wc = 0
    for sent in sentences:
        words = sent.strip().split()
        score = lm.score(sent, bos=True, eos=False)
        word_count = len(words)
        total_wc += word_count
        total_nll += score
    ppl = 10**-(total_nll/total_wc)
    return ppl


class SNLIDataset(data.Dataset):

    def __init__(self, path="./data/classifier", train=True,
                 vocab_size=11000, maxlen=10, reset_vocab=None):
        self.train = train
        self.train_data=[]
        self.test_data=[]
        self.root = path
        self.train_path = os.path.join(path, 'train.txt')
        self.test_path = os.path.join(path, 'test.txt')
        self.lowercase = True
        self.sentence_path =path+"/sentences.dlnlp"
        self.dictionary = Dictionary()
        self.sentence_ids = {}
        self.vocab_size = vocab_size
        self.labels = {'entailment':0, 'neutral':1, 'contradiction':2}
        self.maxlen = maxlen
        
        if reset_vocab:
            self.dictionary.word2idx = deepcopy(reset_vocab)
            self.dictionary.idx2word = {v: k for k, v in self.dictionary.word2idx.items()}
        else:
            self.make_vocab()
            
        if os.path.exists(self.root+"/sent_ids.pkl"):
            self.sentence_ids = pkl.load(open(self.root+"/sent_ids.pkl",'r'))
        else:
            print("Sentence IDs not found!!")
            
        if self.train and os.path.exists(self.train_path):
            self.train_data = self.tokenize(self.train_path)
        if (not self.train) and os.path.exists(self.test_path):
            self.test_data = self.tokenize(self.test_path)

    def __getitem__(self, index):
        if self.train:
            return self.train_data[index]
        else:
            return self.test_data[index]
        
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
        
    def make_vocab(self):
        # Add words to the dictionary
        with open(self.sentence_path, 'r') as f:
            for lines in f:
                toks=lines.strip().split('\t')
                self.sentence_ids[toks[0]]=toks[1].strip()
                line = self.sentence_ids[toks[0]]
                if self.lowercase:
                    # -1 to get rid of \n character
                    words = line[:-1].lower().split(" ")
                else:
                    words = line[:-1].split(" ")
                for word in words:
                    #word = word.decode('Windows-1252').encode('utf-8')
                    self.dictionary.add_word(word)

        # prune the vocabulary
        self.dictionary.prune_vocab(k=self.vocab_size, cnt=False)
        pkl.dump(self.sentence_ids, open(self.root+"/sent_ids.pkl", 'w'))
        pkl.dump(self.dictionary.word2idx, open(self.root+"/vocab_"+str(len(self.dictionary.word2idx))+".pkl", 'w'))

    def get_indices(self, words):
        vocab = self.dictionary.word2idx
        unk_idx = vocab['<oov>']
        return torch.LongTensor([[vocab[w] if w in vocab else unk_idx for w in words]])
    
    def tokenize(self, path):
        """Tokenizes a text file."""
        dropped = 0
        with open(path, 'r') as f:
            linecount = 0
            lines = []
            for line in f:
                linecount += 1
                tokens = line.strip().split('\t')
                label = self.labels[tokens[0]]
                premise = self.sentence_ids[tokens[1]]
                hypothesis = self.sentence_ids[tokens[2]]
                
                if self.lowercase:
                    hypothesis = hypothesis.strip().lower()
                    premise = premise.strip().lower()
                    
                premise_words = premise.strip().split(" ")
                hypothesis_words = hypothesis.strip().split(" ")
                premise_words = ['<sos>'] + premise_words
                premise_words += ['<eos>']
                hypothesis_words = ['<sos>'] + hypothesis_words
                #hypothesis_words += ['<eos>']
                    
                if ((len(premise_words) > self.maxlen+1) or \
                    (len(hypothesis_words) > self.maxlen)):
                    dropped += 1
                    continue
                               
                # vectorize
                vocab = self.dictionary.word2idx
                unk_idx = vocab['<oov>']
                hypothesis_indices = [vocab[w] if w in vocab else unk_idx for w in hypothesis_words]
                premise_indices = [vocab[w] if w in vocab else unk_idx for w in premise_words]
                premise_words = [w if w in vocab else '<oov>' for w in premise_words]
                hypothesis_words = [w if w in vocab else '<oov>' for w in hypothesis_words]
                hypothesis_length = min(len(hypothesis_words), self.maxlen)
                #hypothesis_length = max(hypothesis_length, self.maxlen)
                
                if len(premise_indices) < self.maxlen:
                    premise_indices += [0]*(self.maxlen- len(premise_indices))
                    premise_words += ["<pad>"]*(self.maxlen - len(premise_words))
                    
                if len(hypothesis_indices) < self.maxlen:
                    hypothesis_indices += [0]*(self.maxlen - len(hypothesis_indices))
                    hypothesis_words += ["<pad>"]*(self.maxlen - len(hypothesis_words))
                    
                premise_indices = premise_indices[:self.maxlen]
                hypothesis_indices = hypothesis_indices[:self.maxlen]
                premise_words = premise_words[:self.maxlen]
                hypothesis_words = hypothesis_words[:self.maxlen]
                
                if self.train:
                    lines.append([premise_indices, hypothesis_indices, label])
                else:
                    lines.append([premise_indices, hypothesis_indices, label,
                                  premise_words, hypothesis_words, hypothesis_length])

        print("Number of sentences dropped from {}: {} out of {} total".
              format(path, dropped, linecount))
        return lines


def load_embeddings(root = './data/classifier/'):
    vocab_path=root+'vocab.pkl'
    file_path=root+'embeddings'
    vocab = pkl.load(open(vocab_path, 'r'))
    
    embeddings = torch.FloatTensor(len(vocab),100).uniform_(-0.1, 0.1)
    embeddings[0].fill_(0)
    embeddings[1].copy_(torch.FloatTensor(
        map(float, open(file_path).read().split('\n')[0].strip().split(" ")[1:])))
    embeddings[2].copy_(embeddings[1])
    
    with open(file_path) as fr:
        for line in fr:
            elements=line.strip().split(" ")
            word = elements[0]
            emb = torch.FloatTensor(map(float, elements[1:]))
            if word in vocab:
                embeddings[vocab[word]].copy_(emb)
            
    return embeddings


def get_delta(tensor, right, z_range, gpu):
    bs, dim = tensor.size()
    num_sample = bs*dim
    tensor1 = np.random.uniform(-1*right, -1*right+z_range, num_sample).tolist()
    tensor2 = np.random.uniform(right-z_range,right,  num_sample).tolist()
    samples_delta_z = list(set(tensor1).union(set(tensor2)))
    random.shuffle(samples_delta_z)
    samples_delta_z = to_gpu(gpu,torch.FloatTensor(samples_delta_z[:num_sample]).view(bs, dim))
    return samples_delta_z


def collate_snli(batch):
    premise=[]
    hypothesis=[]
    labels =[]
    lengths = []
    premise_words = []
    hypothesis_words = []
    
    if len(batch[0]) == 3:
        for b in batch:
            x, y, z  = b
            premise.append(x)
            hypothesis.append(y)
            labels.append(z)
        return Variable(torch.LongTensor(premise)), Variable(torch.LongTensor(hypothesis)), Variable(torch.LongTensor(labels)) 
    elif len(batch[0]) == 6:
        for b in batch:
            x, y, z , p, h, l = b
            premise.append(x)
            hypothesis.append(y)
            labels.append(z)
            lengths.append(l)
            premise_words.append(p)
            hypothesis_words.append(h)
            
        return Variable(torch.LongTensor(premise)), Variable(torch.LongTensor(hypothesis)), \
                    Variable(torch.LongTensor(labels)) , premise_words,hypothesis_words , lengths      
    else:
        print("sentence length doesn't match")
