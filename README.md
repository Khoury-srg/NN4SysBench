# NN4SysBench: Characterizing Neural Network Verification for Computer Systems
## Introduction
We propose a benchmark suite for neural network verification for systems 
(NN4Sys) in this repository. This suite includes verification 
benchmark for learned index, learned cardinality, learned Internet congestion
control, learned adaptive bitrate, learned distributed system scheduler, 
which are five tasks that apply neural networks to solve traditional tasks for systems. 


## Instructions to reproduce the result
### Train the model
Code for training the model are put in /Models directory. We provide training script for learned Internet congestion
control, learned adaptive bitrate and learned distributed system scheduler. Detailed instruction 
can be found under each model directory. 
Trained models are provided, so you can skip this step.

### Create fixed input
You will need to install necessary dependencies.
To generate fixed input files for models, run
```
cd Models
python gen_upper.py --model {"pensieve", "decima", "lindex", "cardinality", "bloom_filter", "aurora", "all"}
cd ..
```
You can skip this step as these files are provided in our repo.

### Create onnx models
To create onnx models from trained models, run
```
cd Models
python export.py --model {"pensieve", "decima", "lindex", "cardinality", "bloom_filter", "aurora", "all"}
cd ..
```
You can skip this step as onnx models are provided in our repo. Models for Cardinality are too large to put in github, you can download it from https://drive.google.com/drive/folders/1p_U812OnrMBjQpn5YHhldvHbUmKfnOuh?usp=drive_link and put them under Benchmark/onnx/

### Create specifications
To crete specifications, run
```
cd Benchmarks
python generate_properties.py {random seed, default is 2024}
cd ..
```

### Verify with alpha-beta-crown
install abcrown https://github.com/Verified-Intelligence/alpha-beta-CROWN
run
```
cd Verification
python abcrown_run.py --path {abcrown.py path} --model {"pensieve", "decima", "lindex", "cardinality", "bloom_filter", "aurora", "all"}
cd ..
```

### Verify with marabou
Install marabou https://github.com/NeuralNetworkVerification/Marabou/tree/master
run
```
cd Verification
python marabou_run.py --path {runMarabou.py path} --model {"pensieve", "decima", "lindex", "cardinality", "bloom_filter", "aurora", "all"}
cd ..
```

### Draw figures
run
```
cd Verification/figures
python create_json.py
python draw.py
cd ../..
```