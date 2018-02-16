## Experiment on Images

Train the framework for generation using `mnist_wgan_inv.py`

or use pre-trained framework located in `./models`

Then generate natural adversaries using `mnist_natural_adversary.py`

Classifiers: 
- Random Forest (90.45%), `--classifier rf`
- LeNet (98.71%), `--classifier lenet`

Algorithms: 
- iterative stochastic search, `--iterative` 
- hybrid shrinking search (default)

Output samples are located in `./examples`

`./tflib` is based on [Gulrajani et al., 2017](https://github.com/igul222/improved_wgan_training)