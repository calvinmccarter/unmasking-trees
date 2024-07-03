# unmasking-trees 😷➡️🥳 🌲🌲🌲

[![PyPI version](https://badge.fury.io/py/utrees.svg)](https://badge.fury.io/py/utrees)

UnmaskingTrees is a method for tabular data generation and imputation. It's an order-agnostic autoregressive diffusion model, wherein a training dataset is contruscted by incrementally masking features in random order. Per-feature gradient-boosted trees are then trained to unmask each feature. To better model conditional distributions which are multi-modal ("modal" as in "mode", not as in "modality"), we by default discretize continuous features into bins. 

At inference time, these trees are applied in random order. For generation tasks, these are applied to all features; for imputation tasks, these are applied to features with missing values. This method injects randomness in both data generation and multiple-imputation, from three sources. First, we randomly generate the order over features in which we apply the tree models. Second, we do not "greedily decode" the most likely bin, but instead sample according to predicted probabilities, via nucleus sampling. Third, for continuous features, having sampled a particular bin, we sample from within the bin, treating it as a uniform distribution.

## Installation 

### Installation from PyPI
```
pip install utrees
```

### Installation from source
After cloning this repo, install the dependencies on the command-line, then install utrees:
```
pip install -r requirements.txt
pip install -e .
pytest
```

## Usage

### Synthetic data generation

You can fit `utrees.UnmaskingTrees` the way you would an sklearn model, with the added option that you can call `fit` with `quantize_cols`, a list of bools to specify which columns are continuous (and therefore need to be discretized). By default, all columns are assumed to contain continuous features.

```
import numpy as np
from sklearn.datasets import make_moons
from utrees import UnmaskingTrees
data, labels = make_moons((100, 100), shuffle=False, noise=0.1, random_state=123)  # size (200, 2)
utree = UnmaskingTrees().fit(data)
```

Then, you can generate new data:

```
newdata = utree.generate(n_generate=123)  # size (123, 2)
```

### Missing data imputation

You can fit your `UnmaskingTrees` model on data with missing elements, provided as `np.nan`. You can then impute the missing values, potentially with multiple imputations per missing element. Given an array of `(n_samples, n_dims)`, you will get back an array of size `(n_impute, n_samples, n_dims)`, where the NaNs have been replaced while the others are unchanged.

```
data4impute = data.copy()
data4impute[:, 1] = np.nan
X=np.concatenate([data, data4impute], axis=0)  # size (400, 2)
utree = UnmaskingTrees().fit(X)                                                                                    
imputeddata = utree.impute(n_impute=5)  # size (5, 400, 2)
```

You can also provide a totally new dataset to be imputed, so the model performs imputation without retraining:

```
utree = UnmaskingTrees().fit(data)                                                                                    
imputeddata = utree.impute(n_impute=5, X=data4impute)  # size (5, 200, 2)
```

### Hyperparameters

- n_bins: Number of bins for discretizing continuous features.
- duplicate_K: Number of random masking orders per actual sample. The training dataset will be of size `(n_samples * n_dims * duplicate_K, n_dims)`.
- top_p: Nucleus sampling parameter for inference.
- xgboost_kwargs: dict to pass to XGBClassifier.
- random_state: controls randomness.

## Citing this method

Please consider citing UnmaskingTrees as TODO. 

Also, please consider citing ForestDiffusion ([code](https://github.com/SamsungSAILMontreal/ForestDiffusion) and [paper](https://arxiv.org/abs/2309.09968)), which this work builds on.

