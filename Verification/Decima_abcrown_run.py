import os
import sys

os.environ['MKL_THREADING_LAYER'] = 'GNU'

MODEL_TYPES = ['simple', 'simple', 'concat']
MODEL_SIZES = ['mid']
SIZES = [10, 10, 10]
SPEC_TYPES = [1, 2]

# create yaml
vnn_dir_path = '../Benchmarks/vnnlib'
onnx_dir_path = '../Benchmarks/onnx'
yaml_path = './decima_yaml'
running_result_path = './decima_abcrown_running_result'
timeout = 100
csv_data = []
total_num = 0
current_gpu = 0
P_RANGE = [1, 2, 3, 4, 5]

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
            "solver:\n  batch_size: 1\nbab:\n  branching:\n    method: sb\n    sb_coeff_thresh: 0.1\n    input_split:\n      enable: True")


def main(abcrown_path):
    for i in range(len(SPEC_TYPES)):
        for size in range(SIZES[i]):
            for MODEL in MODEL_SIZES:

                vnn_path = vnn_dir_path + '/decima_' + str(SPEC_TYPES[i]) + f'_' + str(size) + '.vnnlib'
                onnx_path = onnx_dir_path + '/decima_mid_' + MODEL_TYPES[i] + '.onnx'
                yaml = yaml_path + '/decima_' + str(SPEC_TYPES[i]) + f'_' + str(size) + '.yaml'
                create_yaml(yaml, vnn_path, onnx_path)
                os.system(
                    f"python {abcrown_path} --config {yaml} | tee {running_result_path}/decima_{MODEL}_{SPEC_TYPES[i]}_{size}.txt")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: Decima_abcorwn_run.py abcrown_path")
        exit(1)
    abcrown_path = sys.argv[1]
    main(abcrown_path)
