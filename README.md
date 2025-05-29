## SGD as Free Energy Minimization: A Thermodynamic View on Neural Network Training

In this repository, we provide code for reproducing the experiments and visualizations from our paper.
Folder `free_energy` contains environment for training scale-invariant neural networks,
as well as jupyter notebooks for temperature and free energy estimation.
Folder `toy` contains jupyter notebooks, dedicated to our toy example on 3D sphere.

## Abstract 

We present a thermodynamic interpretation of the stationary behavior of stochastic gradient descent (SGD) under fixed learning rates (LRs) in neural network training.
We show that SGD implicitly minimizes a free energy function $F=U-TS$, balancing training loss $U$ and the entropy of the weights distribution $S$, with temperature $T$ determined by the LR.
This perspective offers a new lens on why high LRs prevent training from converging to the loss minima and how different LRs lead to stabilization at different loss levels.
We empirically validate the free energy framework on both underparameterized (UP) and overparameterized (OP) models.
UP models consistently follow free energy minimization, with temperature increasing monotonically with LR, while for OP models, the temperature effectively drops to zero at low LRs, causing SGD to minimize the loss directly and converge to an optimum.
We attribute this mismatch to differences in the signal-to-noise ratio of stochastic gradients near optima, supported by both a toy example and neural network experiments.

## Code

### Installation
```bash
pip install -r requirements.txt
```
### Usage
To train an underparametrized ConvNet on CIFAR-10 with learning rate 0.01:
```bash
cd free_energy
python launch.py
```

### Parameters
To replicate other results from the paper, vary the parameters in the same script:
- dataset: cifar10 or cifar100
- model: convnet or resnet
- num_iters: 1_600_000 for ConvNet models and 1_000_000 for ResNet-18 models
- num_channels: 
    - 8/64 for UP/OP ConvNet on CIFAR-10
    - 8/96 for UP/OP ConvNet on CIFAR-100
    - 4/32 for UP/OP ResNet-18 on CIFAR-10
    - 4/48 for UP/OP ResNet-18 on CIFAR-100
- samples_per_label: 5000 for CIFAR-10, 500 for CIFAR-100.
    - To replicate experiment with different subset sizes, use the following values: [500, 1250, 2500, 5000].

### Extra
Additionally, we provide a script for calculating expected loss and entropy for uniformly
distributed weights on the unit sphere in `free_energy/uniform_metrics.py`.