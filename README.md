# Drift Diffusion Fitting Example

This repository provides example code for fitting Ratcliff’s Drift Diffusion Model (DDM) to human behavioral data using three different inference approaches.

The target task is a standard two-alternative forced-choice (2AFC) reaction-time paradigm, in which a participant must press the correct button as quickly as possible when one of two lamps is illuminated.

## Implemented Fitting Methods

1. **Maximum Likelihood Estimation (MLE)**
2. **Approximate Bayesian Computation (ABC)**
3. **Amortized Inference (AMI)**

Each method demonstrates a different strategy for estimating core DDM parameters (e.g., drift rate, boundary separation, non-decision time) from empirical response time and choice data.

## References

- **Drift Diffusion Model (Ratcliff et al.)**: [link to reference](10.1162/neco.2008.12-06-420)
- **Approximate Bayesian Computation (ABC)**: [link to reference](https://doi.org/10.1145/3025453.3025576)
- **Amortized Inference (AMI)**: [link to reference](https://doi.org/10.1145/3544548.3581439)

## Credits

- Overall implementation (except items below): **June-Seop Yoon**
- MLE implementation using `pymoo`: **Namsub Kim**
- ABC baseline implementation and empirical dataset: **Jonghyun Kim**
- Original Amortized Inference implementation: **Hee-seung Moon**
