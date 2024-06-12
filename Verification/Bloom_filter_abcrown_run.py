import os
import sys

os.environ['MKL_THREADING_LAYER'] = 'GNU'

MODELS = ['small', 'mid', 'big']
MODEL_TYPES = ['simple', 'simple', 'simple', 'parallel', 'concat']
DIFFICULTY = ['easy']
SIZES = [10, 10, 10, 10, 10]
SIZE = 10

SPEC_TYPES = [101, 102, 2, 3, 4]
SPEC_ARRAY_LENGTH = [30, 30, 30, 60, 150]
SPEC_ARRAY_NUM = 3000
HISTORY = 10


# create yaml
vnn_dir_path = '../Benchmarks/vnnlib'
onnx_dir_path = '../Benchmarks/onnx'
yaml_path = './bloom_filter_yaml'
running_result_path = './bloom_filter_abcrown_running_result'
timeout = 100
csv_data = []
total_num = 0
current_gpu = 0

if not os.path.exists(running_result_path):
    os.makedirs(running_result_path)
if not os.path.exists(yaml_path):
    os.makedirs(yaml_path)

def create_yaml(yaml, vnn_path, onnx_path, inputshape=6):
    with open(yaml, mode='w') as f:
        f.write("general:\n  enable_incomplete_verification: False\n  conv_mode: matrix\n")
        f.write(f'model:\n  onnx_path: {onnx_path}\n')
        f.write(f'specification:\n  vnnlib_path: {vnn_path}\n')
        f.write(
            "solver:\n  batch_size: 2048\nbab:\n  branching:\n    method: sb\n    sb_coeff_thresh: 0.1\n    input_split:\n      enable: True")


def main(abcrown_path):
    for i in range(SIZE):
        vnn_path = f'{vnn_dir_path}/bloom_filter_{i}.vnnlib'
        onnx_path = onnx_dir_path + '/bloom_filter.onnx'
        yaml = yaml_path + f'/bloom_filter_{i}.yaml'
        create_yaml(yaml, vnn_path, onnx_path)
        os.system(f"python {abcrown_path} --config {yaml} | tee {running_result_path}/bloom_filter_{i}.txt")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: bloom_filter_abcorwn_run.py abcrown_path")
        exit(1)
    abcrown_path = sys.argv[1]
    main(abcrown_path)
