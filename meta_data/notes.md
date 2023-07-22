# GAN

## Problems
- imbalance
- privacy
- ...
 
## Measuring quality/ objecties
- **Utility**: as motivation to generated data is to improve performance in ML tasks, we may measure how the synethesized data performs dimension-wise prediction as better than without synthesis 
- **attribute fidelity**: generated attributes are faithful to original attributes
- **column-pair fidelity**: measures the discrepancy between the correlation matrices of the real ($X$) and synthetic data ($Z$)
- **marginal fidelity**: marginals of $z_i$ align with the marginals of $x_i$
  - Jensen-Shannon divergence, Wasserstein distance, column correlation, histogram intersection

    
## Formulation 

$\mathcal D$ seeks to minimize 
the **Wasserstein distance** between real $x$ and generated data $z$ while the $\mathcal G$ seeks to maximize the same distance via:

$$E_{z ~ p_{gen}}[ \mathcal{D}(\mathcal{G(z)}) ] - E_{x ~ p_{data}}[ \mathcal{D(x)} ] +\lambda || \triangledown_x D(\hat{x}) ||_2 - 1 $$

- Term 1
- Term 2
- Term 3: gradient penalty

**TableGAN** (Park et al., 2018)

**CTGAN** (Xu et al., 2019): 
+ high attribute fidelity 
- problem when low-data

**TVAE** (Xu et al., 2019) 

**MargCTGAN** (Afonja et al., 2023)
> feature matching of decorrelated marginals in the principal componenet space
