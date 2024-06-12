# Pensieve
Implement pensieve in Pytorch


### Prerequisites
- pytorch 1.13.1, python 3.9, matplotlib, numpy, (MacOS M1)

### Training
- To train a new model, change "MODEL" in train.py to choose different models, run
```
python train.py
```

### Testing
- To test the trained model in simulated environment, change the `NN_MODEL` field of `test.py` , then run 
```
python test.py
```