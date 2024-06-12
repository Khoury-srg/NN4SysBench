import os
import sys

os.environ['MKL_THREADING_LAYER'] = 'GNU'

MODEL_NAMES=["lindex","lindex_deep"]
SIZE=10


# create yaml
vnn_dir_path = '../Benchmarks/vnnlib'
onnx_dir_path = '../Benchmarks/onnx'
yaml_path = './lindex_yaml'
running_result_path = './lindex_abcrown_running_result'
timeout = 100
csv_data = []
total_num = 0
current_gpu = 0

if not os.path.exists(running_result_path):
    os.makedirs(running_result_path)
if not os.path.exists(yaml_path):
    os.makedirs(yaml_path)

def create_yaml(yaml, vnn_path, onnx_path):
    with open(yaml, mode='w') as f:

        f.write(f"general:\n  enable_incomplete_verification: False\n  loss_reduction_func: max\n  conv_mode: matrix\nmodel:\n  onnx_path: {onnx_path}\n")
        f.write(f"specification:\n  vnnlib_path: {vnn_path}\nsolver:\n  "
                f"batch_size: 128  # Number of parallel domains to compute on GPU.\n  bound_prop_method: forward+backward\n  beta-crown:\n"
                f"    iteration: 10  # Iterations for computing intermediate layer bounds.\n")
        f.write(f"bab:\n  initial_max_domains: 1000\n  branching:\n    method: naive  # Split on input space.\n    input_split:\n"
                f"      enable: True\n      adv_check: .inf\n")
        #f.write(f"attack:\n  pgd_order: skip")





def main(abcrown_path):
    for i in range(SIZE):
        for model in MODEL_NAMES:
            for dif in range(3):
                vnn_path = f'{vnn_dir_path}/lindex_{dif}_{i}.vnnlib'
                onnx_path =  f'{onnx_dir_path}/{model}.onnx'
                yaml = yaml_path + f'/{model}_{i}.yaml'
                create_yaml(yaml, vnn_path, onnx_path)
                os.system(f"python {abcrown_path} --config {yaml} | tee {running_result_path}/{model}_{dif}_{i}.txt")



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: Lindex_abcorwn_run.py abcrown_path")
        exit(1)
    abcrown_path = sys.argv[1]
    main(abcrown_path)
