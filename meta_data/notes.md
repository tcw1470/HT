# GAN

## Problems
- imbalance
- privacy
- 
## Formulation 

$\mathcal D$ seeks to minimize 
the **Wasserstein distance** between real $x$ and generated data $z$ while the $\mathcal G$ seeks to maximize the same distance via:

$\min\max \mathbb{E}_{z ~ p_{gen}}[ \mathcal{D}(\mathcal{G(z)}) ] - \mathbb{E}_{x ~ p_{data}}[ \mathcal{D(x)} ] +\lambda || \triangledown_x D(\hat{x}) ||_2 - 1 $


- Term 1
- Term 2
- Term 3: gradient penalty  
## Measuring quality
- attribute fidelity: synthesized attributes
-  
**TableGAN** (Park et al., 2018)

**CTGAN** (Xu et al., 2019): 
+ high attribute fidelity 
- problem when low-data

**TVAE** (Xu et al., 2019) 

**MargCTGAN** (Afonja et al., 2023)
> feature matching of decorrelated marginals in the principal componenet space
