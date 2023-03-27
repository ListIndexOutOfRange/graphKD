# Graph Knowledge Distillation


## Baseline

- **teacher**: resnet 110 | first feature maps: 64 | large = False (27,576,384 params)
- **student**: resnet 20 | first feature maps: 64 | large = False (4,325,184 params)

![](./assets/baseline_cifar100.png)


Le student underfit, il faudrait prendre un truc un poil meilleur mais les trains sont longs, donc peut être plutôt
prendre un teacher plus faible et faire sur cifar10 (au moins pour dev).


## Distance

- pas de problème pour calculer des cosine similarities intra layer
- pas de problème pour utiliser POT mais pas facile d'avoir un truc différentiable; c.f:
    - [Optimizing the Gromov-Wasserstein distance with PyTorch](https://pythonot.github.io/master/auto_examples/backends/plot_optim_gromov_pytorch.html)
    - [Gromov and Fused-Gromov-Wasserstein](https://pythonot.github.io/master/auto_examples/index.html#gromov-and-fused-gromov-wasserstein)