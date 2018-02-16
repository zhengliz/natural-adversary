import numpy as np


def iterative_search(gen_fn, inv_fn, cla_fn, x, y, y_t=None, z=None,
                     nsamples=5000, step=0.01, l=0., h=10., p=2, verbose=False):
    """
    Algorithm 1 in the paper, iterative stochastic search
    :param gen_fn: function of generator, G_theta
    :param inv_fn: function of inverter, I_gamma
    :param cla_fn: function of classifier, f
    :param x: input instance
    :param y: label
    :param y_t: target label for adversary
    :param z: latent vector corresponding to x
    :param nsamples: number of samples in each search iteration
    :param step: Delta r for search step size
    :param l: lower bound of search range
    :param h: upper bound of search range
    :param p: indicating norm order
    :param verbose: print out
    :return: adversary for x against cla_fn (d_adv is Delta z between z and z_adv)
    """
    x_adv, y_adv, z_adv, d_adv = None, None, None, None
    h = l + step

    def printout():
        if verbose and y_t is None:
            print("UNTARGET y=%d y_adv=%d d_adv=%.4f l=%.4f h=%.4f" % (y, y_adv, d_adv, l, h))
        elif verbose:
            print("TARGETED y=%d y_adv=%d d_adv=%.4f l=%.4f h=%.4f" % (y, y_adv, d_adv, l, h))

    if verbose:
        print("iterative search")

    if z is None:
        z = inv_fn(x)

    while True:
        delta_z = np.random.randn(nsamples, z.shape[1])     # http://mathworld.wolfram.com/HyperspherePointPicking.html
        d = np.random.rand(nsamples) * (h - l) + l          # length range [l, h)
        norm_p = np.linalg.norm(delta_z, ord=p, axis=1)
        d_norm = np.divide(d, norm_p).reshape(-1, 1)        # rescale/normalize factor
        delta_z = np.multiply(delta_z, d_norm)
        z_tilde = z + delta_z       # z tilde
        x_tilde = gen_fn(z_tilde)   # x tilde
        y_tilde = cla_fn(x_tilde)   # y tilde

        if y_t is None:
            indices_adv = np.where(y_tilde != y)[0]
        else:
            indices_adv = np.where(y_tilde == y_t)[0]

        if len(indices_adv) == 0:       # no candidate generated
            l = h
            h = l + step
        else:                           # certain candidates generated
            idx_adv = indices_adv[np.argmin(d[indices_adv])]

            if y_t is None:
                assert (y_tilde[idx_adv] != y)
            else:
                assert (y_tilde[idx_adv] == y_t)

            if d_adv is None or d[idx_adv] < d_adv:
                x_adv = x_tilde[idx_adv]
                y_adv = y_tilde[idx_adv]
                z_adv = z_tilde[idx_adv]
                d_adv = d[idx_adv]
                printout()
                break

    adversary = {'x': x, 'y': y, 'z': z,
                 'x_adv': x_adv, 'y_adv': y_adv, 'z_adv': z_adv, 'd_adv': d_adv}

    return adversary


def recursive_search(gen_fn, inv_fn, cla_fn, x, y, y_t=None, z=None,
                     nsamples=5000, step=0.01, l=0., h=10., stop=5, p=2, verbose=False):
    """
    Algorithm 2 in the paper, hybrid shrinking search
    :param gen_fn: function of generator, G_theta
    :param inv_fn: function of inverter, I_gamma
    :param cla_fn: function of classifier, f
    :param x: input instance
    :param y: label
    :param y_t: target label for adversary
    :param z: latent vector corresponding to x
    :param nsamples: number of samples in each search iteration
    :param step: Delta r for search step size
    :param l: lower bound of search range
    :param h: upper bound of search range
    :param stop: budget of extra iterative steps
    :param p: indicating norm order
    :param verbose: print out
    :return: adversary for x against cla_fn (d_adv is Delta z between z and z_adv)
    """
    x_adv, y_adv, z_adv, d_adv = None, None, None, None
    counter = 1

    def printout():
        if verbose and y_t is None:
            print("UNTARGET y=%d y_adv=%d d_adv=%.4f l=%.4f h=%.4f count=%d" % (y, y_adv, d_adv, l, h, counter))
        elif verbose:
            print("TARGETED y=%d y_adv=%d d_adv=%.4f l=%.4f h=%.4f count=%d" % (y, y_adv, d_adv, l, h, counter))

    if verbose:
        print("first recursion")

    if z is None:
        z = inv_fn(x)

    while True:
        delta_z = np.random.randn(nsamples, z.shape[1])     # http://mathworld.wolfram.com/HyperspherePointPicking.html
        d = np.random.rand(nsamples) * (h - l) + l          # length range [l, h)
        norm_p = np.linalg.norm(delta_z, ord=p, axis=1)
        d_norm = np.divide(d, norm_p).reshape(-1, 1)        # rescale/normalize factor
        delta_z = np.multiply(delta_z, d_norm)
        z_tilde = z + delta_z       # z tilde
        x_tilde = gen_fn(z_tilde)   # x tilde
        y_tilde = cla_fn(x_tilde)   # y tilde

        if y_t is None:
            indices_adv = np.where(y_tilde != y)[0]
        else:
            indices_adv = np.where(y_tilde == y_t)[0]

        if len(indices_adv) == 0:       # no candidate generated
            if h - l < step:
                break
            else:
                l = l + (h - l) * 0.5
                counter = 1
                printout()
        else:                           # certain candidates generated
            idx_adv = indices_adv[np.argmin(d[indices_adv])]

            if y_t is None:
                assert (y_tilde[idx_adv] != y)
            else:
                assert (y_tilde[idx_adv] == y_t)

            if d_adv is None or d[idx_adv] < d_adv:
                x_adv = x_tilde[idx_adv]
                y_adv = y_tilde[idx_adv]
                z_adv = z_tilde[idx_adv]
                d_adv = d[idx_adv]
                l, h = d_adv * 0.5, d_adv
                counter = 1
            else:
                h = l + (h - l) * 0.5
                counter += 1

            printout()
            if counter > stop or h - l < step:
                break

    if verbose:
        print('then iteration')

    if d_adv is not None:
        h = d_adv
    l = max(0., h - step)
    counter = 1
    printout()

    while counter <= stop and h > 1e-4:
        delta_z = np.random.randn(nsamples, z.shape[1])
        d = np.random.rand(nsamples) * (h - l) + l
        norm_p = np.linalg.norm(delta_z, ord=p, axis=1)
        d_norm = np.divide(d, norm_p).reshape(-1, 1)
        delta_z = np.multiply(delta_z, d_norm)
        z_tilde = z + delta_z
        x_tilde = gen_fn(z_tilde)
        y_tilde = cla_fn(x_tilde)

        if y_t is None:
            indices_adv = np.where(y_tilde != y)[0]
        else:
            indices_adv = np.where(y_tilde == y_t)[0]

        if len(indices_adv) == 0:
            counter += 1
            printout()
        else:
            idx_adv = indices_adv[np.argmin(d[indices_adv])]

            if y_t is None:
                assert (y_tilde[idx_adv] != y)
            else:
                assert (y_tilde[idx_adv] == y_t)

            if d_adv is None or d[idx_adv] < d_adv:
                x_adv = x_tilde[idx_adv]
                y_adv = y_tilde[idx_adv]
                z_adv = z_tilde[idx_adv]
                d_adv = d[idx_adv]

            h = l
            l = max(0., h - step)
            counter = 1
            printout()

    adversary = {'x': x, 'y': y, 'z': z,
                 'x_adv': x_adv, 'y_adv': y_adv, 'z_adv': z_adv, 'd_adv': d_adv}

    return adversary


if __name__ == '__main__':
    pass
