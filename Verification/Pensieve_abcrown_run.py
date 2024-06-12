import os
import sys

os.environ['MKL_THREADING_LAYER'] = 'GNU'

MODEL_TYPES = ['simple', 'simple', 'parallel']
MODEL_SIZES = ['small', 'mid', 'big']
P_RANGE = [0.05, 0.1, 0.5, 0.7, 1]
SIZES = [10, 10, 10]
SIZE=10
SPEC_TYPES = [1, 2, 3]


# create yaml
vnn_dir_path = '../Benchmarks/vnnlib'
onnx_dir_path = '../Benchmarks/onnx'
yaml_path = './pensieve_yaml'
running_result_path = './pensieve_abcrown_running_result'
timeout = 100
csv_data = []
total_num = 0
current_gpu = 0
DIMENSION_NUMBERS=[1,2,3]

if not os.path.exists(running_result_path):
    os.makedirs(running_result_path)
if not os.path.exists(yaml_path):
    os.makedirs(yaml_path)


def create_yaml(yaml, vnn_path, onnx_path, inputshape=6):
    with open(yaml, mode='w') as f:
        f.write("general:\n  enable_incomplete_verification: False\n  conv_mode: matrix\n")
        f.write(f'model:\n  onnx_path: {onnx_path}\n')
        f.write('  onnx_quirks: \"{\'Reshape\': {\'fix_batch_size\': True}}\"\n')
        f.write(f'  input_shape: [-1, {inputshape}, 8]\n')
        f.write(f'specification:\n  vnnlib_path: {vnn_path}\n')
        f.write(
            "solver:\n  batch_size: 1\nbab:\n  branching:\n    method: sb\n    sb_coeff_thresh: 0.1\n    input_split:\n      enable: True")


def main(abcrown_path):
    for i in range(len(SPEC_TYPES)):
        MODEL_TYPE = MODEL_TYPES[i]

        for range_ptr in range(len(P_RANGE)):
            for d_ptr in range(len(DIMENSION_NUMBERS)):
                dimension_number = DIMENSION_NUMBERS[d_ptr]
                for MODEL in MODEL_SIZES:
                    for size in range(SIZE):
                        if dimension_number != 2 or range_ptr != 0:
                            continue

                        vnn_path = f'{vnn_dir_path}/pensieve_{SPEC_TYPES[i]}_{dimension_number}_{range_ptr}_{size}.vnnlib'
                        onnx_path = onnx_dir_path + '/pensieve_' + MODEL + '_' + MODEL_TYPE + '.onnx'
                        yaml = yaml_path + f'/pensieve_{MODEL_TYPE}_{MODEL}_{SPEC_TYPES[i]}_{dimension_number}_{range_ptr}_{size}.yaml'
                        if MODEL_TYPE == 'simple':
                            create_yaml(yaml, vnn_path, onnx_path, 6)

                        if MODEL_TYPE == 'parallel':
                            create_yaml(yaml, vnn_path, onnx_path, 12)
                        command = f"python {abcrown_path} --config {yaml} | tee {running_result_path}/pensieve_{MODEL}_{SPEC_TYPES[i]}_{dimension_number}_{range_ptr}_{size}.txt"
                        print("--------------------------------------------------------------------------->")
                        print(command)
                        print("<---------------------------------------------------------------------------")
                        os.system(command)



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python Pensieve_abcorwn_run.py abcrown_path")
        exit(1)
    abcrown_path = sys.argv[1]
    main(abcrown_path)
