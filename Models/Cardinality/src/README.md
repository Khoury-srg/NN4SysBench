Learned Cardinalities in PyTorch
====

PyTorch implementation of multi-set convolutional networks (MSCNs) to estimate the result sizes of SQL queries [1, 2].

## Requirements

  * PyTorch 1.0
  * Python 3.7

## Usage

```python3 train.py --help```

Example usage:

```python3 train.py --train --predict synthetic```

To train a model with hidden size 50 and test on *synthetic*:

```python3 train.py --queries 100000 --train --predict --save_path models_wo_materialize_50 --hid 50 --epochs 100 synthetic```

## References

[1] [Kipf et al., Learned Cardinalities: Estimating Correlated Joins with Deep Learning, 2018](https://arxiv.org/abs/1809.00677)

[2] [Kipf et al., Estimating Cardinalities with Deep Sketches, 2019](https://arxiv.org/abs/1904.08223)
