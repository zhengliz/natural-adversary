import argparse
from models import Baseline_Embeddings, Baseline_LSTM
from utils import to_gpu, Corpus, batchify, SNLIDataset, collate_snli
import random
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable

def evaluate_model():
    test_iter = iter(testloader)
    correct=0
    total=0
    for batch in test_iter:
        premise, hypothesis, target, _, _, _ = batch
        
        if args.cuda:
            premise=premise.cuda()
            hypothesis = hypothesis.cuda()
            target = target.cuda()
            
        prob_distrib = baseline_model.forward((premise, hypothesis))
        predictions = np.argmax(prob_distrib.data.cpu().numpy(), 1)
        correct+=len(np.where(target.data.cpu().numpy()==predictions)[0])
        total+=premise.size(0)
    acc=correct/float(total)
    print("Accuracy:{0}".format(acc))
    return acc
        
        
parser = argparse.ArgumentParser(description='PyTorch baseline for Text')
parser.add_argument('--data_path', type=str, required=True,
                    help='location of the data corpus')
parser.add_argument('--model_type', type=str, default="emb",
                    help='location of the data corpus')
parser.add_argument('--epochs', type=int, default=20,
                    help='maximum number of epochs')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--packed_rep', type=bool, default=True,
                    help='pad all sentences to fixed maxlen')
parser.add_argument('--train_mode', type=bool, default=True,
                    help='set training mode')
parser.add_argument('--maxlen', type=int, default=10,
                    help='maximum sentence length')
parser.add_argument('--lr', type=float, default=1e-05,
                    help='learning rate')
parser.add_argument('--seed', type=int, default=1111,
                    help='seed')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='beta1 for adam. default=0.9')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use CUDA')
parser.add_argument('--save_path', type=str, required=True,
                    help='used for saving the models')
parser.add_argument('--vocab_size', type=int, default=11004,
                    help='vocabulary size')

args = parser.parse_args()

corpus_train = SNLIDataset(train=True, vocab_size=args.vocab_size, path=args.data_path)
corpus_test = SNLIDataset(train=False, vocab_size=args.vocab_size, path=args.data_path)
trainloader= torch.utils.data.DataLoader(corpus_train, batch_size = args.batch_size, collate_fn=collate_snli, shuffle=True)
train_iter = iter(trainloader)
testloader= torch.utils.data.DataLoader(corpus_test, batch_size = args.batch_size, collate_fn=collate_snli, shuffle=False)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.model_type=="lstm":
    baseline_model = Baseline_LSTM(100,300,maxlen=args.maxlen, gpu=args.cuda)
elif args.model_type=="emb":
    baseline_model = Baseline_Embeddings(100, vocab_size=args.vocab_size)
    
if args.cuda:
    baseline_model = baseline_model.cuda()
optimizer = optim.Adam(baseline_model.parameters(),
                           lr=args.lr,
                           betas=(args.beta1, 0.999))
criterion = nn.CrossEntropyLoss()

best_accuracy = 0
if args.train_mode:
    for epoch in range(0, args.epochs):
        niter = 0
        loss_total = 0
        while niter < len(trainloader):
            niter+=1
            premise, hypothesis, target = train_iter.next()
            if args.cuda:
                premise=premise.cuda()
                hypothesis = hypothesis.cuda()
                target = target.cuda()
            prob_distrib = baseline_model.forward((premise, hypothesis))
            loss=criterion(prob_distrib, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += loss.data[0]
        print(loss_total/float(niter))
        train_iter = iter(trainloader)
        curr_acc = evaluate_model()
        if curr_acc > best_accuracy:
            print("saving model...")
            with open(args.save_path+"/"+args.model_type+'.pt', 'wb') as f:
                torch.save(baseline_model.state_dict(), f)
            best_accuracy = curr_acc
        
    print("Best accuracy :{0}".format(best_accuracy))
