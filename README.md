# Holistic Robust (HR) Neural Networks

<p align="left">
  <img width="150" height="40" src="Misc/python.svg">
   <img width="180" height="40" src="Misc/tf.svg">
    <img width="150" height="40" src="Misc/pt_badge.svg">
</p>

```python
pip install HR_Neural_Networks
```

### This code base is an open-source implementation of the paper ["Certified Robust Neural Networks: Generalization and Corruption Resistance"](https://arxiv.org/pdf/2303.02251.pdf).

Holistic Robust Learning (HR) is a learning approach which provides _certified_ protection against data poisoning and evasion attacks, while enjoying _guaranteed_ generalization. HR minimizes a loss function that is guaranteed to be an upper bound on the out-of-sample performance of the trained networks with high probability. Hence, when training with HR, the out-of-sample performance is at least as good as the observed in-sample performance. This is both guaranteed theoretically and verified empirically.
Robustness is controlled by three parameters: 
* $\alpha$: controls protection against generic data poisoning at training time. This encompasses any kind of corruption in the training data; for instance training examples that have been obscured or which are wholly misspecified. For a given chosen $\alpha$, HR is certified when up to a fraction $\alpha$ of data points are corrupted.
* $\epsilon$: controls protection against small perturbations to the testing or training examples, such as noise or evasive attacks. HR is certified to any adversarial attacks limited to the norm ball { $\delta: ||\delta|| \leq \epsilon$ }. The current implementation supports $\ell_2$ and $\ell_\infty$ balls.
* $r$: controls protection against overfitting to the training instances. The parameter sets the desired strength of generalization and the conservativeness of training. It also reduces variance to randomness of the training data. HR in-sample loss is guaranteed to be an upper bound on the out-of-sample loss with probability $1-e^{-nr +O(1)}$ where $n$ is the data size.

We provide a robust loss function that can be automatically differentiated in Pytorch. If not using Pytorch, we also provide framework-agnostic importance weights that can be integrated with Tensorflow or another deep learning library. Doing so involves minimal disruption to standard training pipelines.

Click here for a **Colab tutorial applying HR for MNIST classification**: 

<p align="left"><a href= "https://colab.research.google.com/drive/1d5BZvCDGWHS_UxFR77YneKGB3mMGR-tY?usp=sharing">
  <img width="247.8" height="42.6" src="Misc/colab.svg"></a>
</p>


## Training of HR Neural Networks in Pytorch

HR can be implemented for neural network training with minimal disruption to typical training pipelines. The core output of the code is a Pytorch loss function which can be optimized via ordinary backpropagation commands. For example, see below for a contrast between regular training and HR training where the difference is basically in one line.
 
### Natural training

```python

criterion = F.cross_entropy(reduction = 'mean')

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()
        
 ```

### HR training
You can install HR simply with a `pip` command

```python
!pip install HR_Neural_Networks
```

Training with the HR loss requires then only to change one line of the training code.

```python

criterion = F.cross_entropy(reduction = 'none') # note the change from mean -> none

########### HR Model Instantiation ###############

from HR_Neural_Networks.HR import * 

α_choice = 0.05 
r_choice = 0.1
ϵ_choice = 0.5
       
HR = HR_Neural_Networks(NN_model = model,
                        train_batch_size = 128,
                        loss_fn = criterion,
                        normalisation_used = None,
                        α_choice = α_choice, 
                        r_choice = r_choice,
                        ϵ_choice = ϵ_choice)

########### Training Loop ###############

def train(HR, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        HR_loss, _ = HR.HR_criterion(inputs = data, targets = target, device = device)

        HR_loss.backward()
        optimizer.step()
```

## Background

HR considers the following setting of learning under corruption: $n$ data points are first sampled from a true clean distribution $\mathcal{D}$ and then less than $\alpha n$ data points are corrupted (poisoning attacks), resulting in an observed corrupted empirical distribution 
$\mathcal{D}\_n$
constituting training data. At test time, samples from $\mathcal{D}$ are perturbed with noise in a set $\mathcal{N}$ (evasive attacks), and the model is tested with distribution $\mathcal{D}\_{\text{test}}$ of perturbed instances.

HR seeks to minimizes an upper bound on the testing loss constructed using the provided corrupted training data. This upper bound–HR loss–is designed using distributionally robust optimization (DRO), by constructing an ambiguity set $\mathcal{U}\_{\mathcal{N}, \alpha, r}(\mathcal{D}\_n)$ around the corrupted empirical distribution $\mathcal{D}\_n$ and optimizing the worst-case loss over distributions realizing in this set. The ambiguity set is constructed to contain the testing distribution $\mathcal{D}\_{\text{test}}$ with high probability. The HR loss writes therefore as

```math
\begin{equation}
\max_{\mathcal{D}' \in \mathcal{U}_{\mathcal{N}, \alpha, r}(\mathcal{D}_n)} \mathbb{E}_{(X, Y) \sim \mathcal{D}'}[\ell(\theta, X, Y)]
\end{equation}
```
where $\ell$ is the given loss function and $\theta$ the network's parameters.

The HR objective function is an upper bound on the test performance with probability $1-e^{-rn+O(1)}$ when less then a fraction $\alpha$ of all $n$ samples are tampered by poisoning, and the evasion corruption is bounded within the set $\mathcal{N}$.
The parameters $\mathcal{N}, r$ and $\alpha$ hence are important design choices and directly reflect the desired robustness. In this implementation, we chose $\mathcal{N} =$ { $\delta: ||\delta|| \leq \epsilon $}.

The HR loss is also proven to be a ``tight'' upper bound. That is, corruption and generalization are efficiently captured and the provided robustness is not overly conservative. In particular, HR captures efficiently the interaction between generalization and corruption. 
For example, when used in conjunction $\mathcal{N}$ and $r$ can provide protection to the well-known phenomenon of  [robust overfitting](https://arxiv.org/abs/2002.11569), where adversarial training exhibit severe overfitting.

## Reference
```
@article{bennouna2023certified,
  title={Certified Robust Neural Networks: Generalization and Corruption Resistance},
  author={Bennouna, Amine and Lucas, Ryan and Van Parys, Bart},
  journal={arXiv preprint arXiv:2303.02251},
  year={2023}
}
```

Please contact amineben@mit.edu and ryanlu@mit.edu if you have any question about the paper or the codes.

