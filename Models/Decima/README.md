# Decima: PyTorch Implementation

Decima is a reinforcement learning-based job scheduling model implemented in PyTorch.

### 1. Install Dependencies
Set up the Conda environment:
```bash
conda env create -f environment.yml
conda activate decima
```

### 2. Training
To train the Decima model, run:
```bash
python train.py
```
The model will be saved periodically in the ```models/``` directory.

### 3. Testing
To evaluate the trained Decima model, run:
```bash
python test.py
```
After testing, the mean reward for the model is printed, which reflects the overall performance of the scheduling decisions.

