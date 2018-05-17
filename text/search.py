import numpy as np
import torch
from copy import deepcopy
from torch.autograd import Variable


def get_min(indices_adv1, indices_adv2, d):
    d1 = deepcopy(d)
    d2 = deepcopy(d)
    idx_adv1 = indices_adv1[np.argmin(d1[indices_adv1])]
    idx_adv2 = indices_adv2[np.argmin(d2[indices_adv2])]
    cnt = 0
    orig_idx_adv1 = idx_adv1
    orig_idx_adv2 = idx_adv2
    while idx_adv1 == idx_adv2 and cnt < 20:
        d1[idx_adv1] = 9999
        d2[idx_adv2] = 9999
        idx_adv1 = indices_adv1[np.argmin(d1[indices_adv1])]
        idx_adv2 = indices_adv2[np.argmin(d2[indices_adv2])]
        cnt+=1
    if cnt == 20:
        return orig_idx_adv1, orig_idx_adv2
    else:
        return idx_adv1, idx_adv2


def search_fast(generator, pred_fn, x, y, z, nsamples=20, right=0.005):
    premise, hypothesis = x
    x_adv1, x_adv2, d_adv1, d_adv2, all_adv = None, None, None, None, None
    right_curr = right
    counter = 0
    while counter<=5: 
        mus = z.repeat(nsamples, 1)
        delta = torch.FloatTensor(mus.size()).uniform_(-1*right_curr, right_curr)
        dist = np.array([np.sqrt(np.sum(x**2)) for x in delta.cpu().numpy()])
        perturb_z = Variable(mus + delta, volatile=True)
        x_tilde = generator(perturb_z)
        y_tilde1, y_tilde2, all_adv = pred_fn((premise, hypothesis, x_tilde, dist))        

        indices_adv1 = np.where(y_tilde1.data.cpu().numpy() != y.data.cpu().numpy())[0]
        indices_adv2 = np.where(y_tilde2.data.cpu().numpy() != y.data.cpu().numpy())[0]

        if (len(indices_adv1)>0) and (indices_adv1[0] == 0):
            indices_adv1 = np.delete(indices_adv1, 0)
        if (len(indices_adv2)>0) and (indices_adv2[0]) == 0:
            indices_adv2 = np.delete(indices_adv2, 0)
            
        if len(indices_adv1) == 0 or len(indices_adv2) == 0:
            counter += 1
            right_curr *= 2
        else:
            idx_adv1, idx_adv2 = get_min(indices_adv1, indices_adv2, dist)
            if d_adv1 is None or ((dist[idx_adv1] < d_adv1) and (dist[idx_adv2] < d_adv2)):
                x_adv1 = x_tilde[idx_adv1]
                x_adv2 = x_tilde[idx_adv2]
                d_adv1 = float(dist[idx_adv1])
                d_adv2 = float(dist[idx_adv2])
            return x_adv1, x_adv2, d_adv1, d_adv2, all_adv

    print("\nGoing into infinite loop.......\n")
    return x_adv1, x_adv2, d_adv1, d_adv2, all_adv

    
def search(generator, pred_fn, x, y, z, nsamples=100, l=0., h=1.0, step=0.005, stop=5, p=2):
    premise, hypothesis = x
    x_adv1, x_adv2, d_adv1, d_adv2, d_adv = None, None, None, None, None
    counter = 1
    loop_cnt = 0
    while True:
        delta_z = np.random.randn(nsamples, z.shape[1]) # http://mathworld.wolfram.com/HyperspherePointPicking.html
        d = np.random.rand(nsamples) * (h - l) + l  # length range [l, h)
        norm_p = np.linalg.norm(delta_z, ord=p, axis=1)
        d_norm = np.divide(d, norm_p).reshape(-1, 1)  # rescale/normalize factor
        delta_z = np.multiply(delta_z, d_norm)
        delta_z += z  # z tilde
        x_tilde = generator(delta_z) # x tilde
        y_tilde1, y_tilde2, all_adv = pred_fn((premise, hypothesis, x_tilde, d))  # y tilde
        
        indices_adv1 = np.where(y_tilde1.data.cpu().numpy() != y.data.cpu().numpy())[0]
        indices_adv2 = np.where(y_tilde2.data.cpu().numpy() != y.data.cpu().numpy())[0]

        if (len(indices_adv1)>0) and (indices_adv1[0] == 0):
            indices_adv1 = np.delete(indices_adv1, 0)
        if (len(indices_adv2)>0) and (indices_adv2[0]) == 0:
            indices_adv2 = np.delete(indices_adv2, 0)
            
        if len(indices_adv1) == 0 or len(indices_adv2) == 0:  # no candidate generated
            if h - l < step:
                break
            else:
                l = l + (h - l) * 0.5
                h += 0.5
        else:  # certain candidates generated
            idx_adv1, idx_adv2 = get_min(indices_adv1, indices_adv2, d)
            if d_adv1 is None or ((d[idx_adv1] < d_adv1) and (d[idx_adv2] < d_adv2)):
                counter = 1
                x_adv1 = x_tilde[idx_adv1]
                x_adv2 = x_tilde[idx_adv2]
                d_adv1 = float(d[idx_adv1])
                d_adv2 = float(d[idx_adv2])
                d_adv = (d_adv1+d_adv2)/2
                l, h = d_adv * 0.5, d_adv
            else:
                counter += 1
                h = l + (h - l) * 0.5
            if counter > stop or h - l < step:
                break

    h = d_adv
    l = max(0., h - step)
    leftward = False
    counter = 1
    while True:
        while (not leftward) and counter <= stop:
            delta_z = np.random.randn(nsamples, z.shape[1])
            d = np.random.rand(nsamples) * (h - l) + l
            norm_p = np.linalg.norm(delta_z, ord=p, axis=1)
            d_norm = np.divide(d, norm_p).reshape(-1, 1)
            delta_z = np.multiply(delta_z, d_norm)
            delta_z += z
            x_tilde = generator(delta_z)
            y_tilde1, y_tilde2, all_adv = pred_fn((premise, hypothesis, x_tilde, d))
            
            indices_adv1 = np.where(y_tilde1.data.cpu().numpy() != y.data.cpu().numpy())[0]
            indices_adv2 = np.where(y_tilde2.data.cpu().numpy() != y.data.cpu().numpy())[0]
    
            if (len(indices_adv1)>0) and (indices_adv1[0] == 0):
                indices_adv1 = np.delete(indices_adv1, 0)
            if (len(indices_adv2)>0) and (indices_adv2[0]) == 0:
                indices_adv2 = np.delete(indices_adv2, 0)
                
            if len(indices_adv1) == 0 or len(indices_adv2) == 0:
                counter += 1
            else:
                counter = 1
                leftward = True
                idx_adv1, idx_adv2 = get_min(indices_adv1, indices_adv2, d)
                if d_adv1 is None or ((d[idx_adv1] < d_adv1) and (d[idx_adv2] < d_adv2)):
                    x_adv1 = x_tilde[idx_adv1]
                    x_adv2 = x_tilde[idx_adv2]
                    d_adv1 = float(d[idx_adv1])
                    d_adv2 = float(d[idx_adv2])
                    
        if not leftward or l < 1e-4:
            break

        leftward = False
        h = l
        l = max(0., h - step)
        if loop_cnt == 10:
            print("\nGoing into infinite loop.......\n")
            return x_adv1, x_adv2, d_adv1, d_adv2, all_adv
        else:
            loop_cnt += 1

    return x_adv1, x_adv2, d_adv1, d_adv2, all_adv
