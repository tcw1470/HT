# GAN

## Problems
- imbalance
- privacy
- ...
 
## Measuring quality/ objecties
- **attribute fidelity**
- **column-pair fidelity**: measures the discrepancy between the correlation matrices of the real ($X$) and synthetic data ($Z$)
- **joint fidelity**: joint distributions of $X$ and $Z$ are similar
- **marginal fidelity**: marginals of $z_i$ align with the marginals of $x_j$
  - Jensen-Shannon divergence, Wasserstein distance, column correlation, histogram intersection
- **Utility**: as motivation to generated data is to improve performance in ML tasks, we may measure how use of $Z$ improves **dimension-wise prediction** when compared to use of $X$ alone

    
## Formulation 

- $\mathcal G$ generates realistic data sample $z_i$
- $\mathcal D$ tries to discriminate generated data from real samples 
- In one formulation, $\mathcal D$ seeks to minimize the **Wasserstein distance** between real $X$ and generated data $Z$ while the $\mathcal G$ seeks to maximize the same distance via:

$$  E_{z ~ p_{gen}}[ \mathcal{D}(\mathcal{G}(z)) ] - E_{x ~ p_{data}}[ \mathcal{D(x)} ] +\lambda || \triangledown_x D(\hat{x}) ||_2 - 1 $$

- Term 1: low when discriminator guesses generated data points $z_i$ incorrectly 
- Term 2: low when discriminator identifies real datapoints $x_j$  correctly
- Term 3: gradient penalty

## Benchmarks

**TableGAN** (Park et al., 2018)
-

**CTGAN** (Xu et al., 2019): 
- high attribute fidelity 
- problem with low-dimensionality 

**TVAE** (Xu et al., 2019) 

**MargCTGAN** (Afonja et al., 2023)
> feature matching of decorrelated marginals in the principal componenet space

$${L_{WGP}+ L_{cond} + L_{marg}$$
