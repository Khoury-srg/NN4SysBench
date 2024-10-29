# Aurora: Reinforcement Learning Resources for Performance-Oriented Congestion Control

Aurora is a reinforcement learning-based project focused on optimizing congestion 
control for network performance. This repository provides resources for training 
and evaluating models in this domain.

## Training Instructions

### 1. Navigate to the Training Directory

Open a terminal and navigate to the `/gym/` directory within the project:

```bash
cd /gym/
```

### 2. Install Required Dependencies

#### Conda Environment Setup

To set up the Conda environment for this project, follow these steps:

1. **Ensure Conda is Installed**

   If Conda is not installed, download and install it from the 
   [official Conda website](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

2. **Create the Conda Environment**

   Use the provided `environment.yml` file to create the environment by running 
   the following command:

   ```bash
   conda env create -f environment.yml
   conda activate aurora
    ```

### 3. Run the Training Script

Once the dependencies are installed, initiate the training by running the 
`train.py` script. You can specify the model type (`small`, `mid`, `big`, or 
`all`) to train different versions of the Aurora model. Use the following 
command:

```bash
python train.py --model {small, mid, big, all}
```
- `small`: Trains the small version of the model.
- `mid`: Trains the mid-sized model.
- `big`: Trains the large version of the model.
- `all`: Trains all three model versions sequentially.

Once training is complete, you will see the following message:
```
[Done] Finished training {model_type} model
```

### 4. Testing
Testing is not provided. Refer to the original paper for the necessary settings.