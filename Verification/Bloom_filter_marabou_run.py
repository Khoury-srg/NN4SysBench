import os
import sys

STATISTIC_RANGE = [0.05, 0.01, 0, 1]
MODEL_SIZES = ['small', 'mid', 'big']
MODEL_TYPES = ['simple', 'simple', 'simple', 'parallel', 'concat']
running_result_path = './bloom_filter_marabou_running_result'
SIZES = [10, 10, 10, 10, 10]
SIZE=10

SPEC_TYPES = [101, 102, 2, 3]
SPEC_ARRAY_LENGTH = [30, 30, 30, 60, 150]
SPEC_ARRAY_NUM = 3000
HISTORY = 10

txt_dir_path = '../Benchmarks/marabou_txt'
onnx_dir_path = '../Benchmarks/onnx'


def main(marabou_path):
    if not os.path.exists(running_result_path):
        os.makedirs(running_result_path)
    for i in range(SIZE):


        command = f'python {marabou_path} {onnx_dir_path}/bloom_filter.onnx {txt_dir_path}/bloom_filter_{i}.txt | tee {running_result_path}/bloom_filter_{i}.txt'
        print(command)
        os.system(command)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: Aurora_marabou_run.py marabou_path")
        exit(1)
    marabou_path = sys.argv[1]
    main(marabou_path)
