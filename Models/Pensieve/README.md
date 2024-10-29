# Pensieve
PyTorch implementation of Pensieve for optimizing video streaming with reinforcement learning.

### 1. Install Dependencies
Set up the Conda environment:
```bash
conda env create -f environment.yml
conda activate pensieve-pytorch
```


### 2. Training
To train a model (`small`, `mid`, `big`), run:
```bash
python train.py --model <model_size>
```
To train all models sequentially, run:
```bash
python train.py --model all
```
The models will be saved at regular intervals in the `./results/` directory.
Note: When the epoch reaches `max_epoch`, the training process will exit, and an error message ("broken pipe") may appear. This is expected behavior and does not indicate an issue.

### 2. Testing
To test a trained model, run:
```bash
python test.py --model <model_size>
```
To test all models sequentially:
```bash
python test.py --model all
```
If you want to test your own model, modify the parameter in the `test.py`.

The testing logs will be saved in the `./test_results/` directory, and the performance of the models will be printed, including the average reward, which reflects the overall quality of the streaming experience (considering video quality, rebuffering, and smoothness).

### 4. Model Options
- **small**: Lightweight model
- **mid**: Medium-sized model
- **big**: High-accuracy model

Test results are saved in the `./test_results/` directory.