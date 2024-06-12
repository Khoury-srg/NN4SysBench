import os
import sys

MODELS = ['small', 'mid', 'big']
MODEL_TYPES = ['simple', 'simple']

running_result_path = './pensieve_marabou_running_result'
SIZES = [10, 10, 10, 10, 10]
P_RANGE = [0.05, 0.1, 0.5, 0.7, 1]

SPEC_TYPES = [1, 2]
DIMENSION_NUMBERS=[1,2,3]

txt_dir_path = '../Benchmarks/marabou_txt'
onnx_dir_path = '../Benchmarks/onnx'

cur_dir = os.getcwd()


def main(marabou_path):
    if not os.path.exists(running_result_path):
        os.makedirs(running_result_path)
    for spec_type in range(len(SPEC_TYPES)):
        for range_ptr in range(len(P_RANGE)):
            for d_ptr in range(len(DIMENSION_NUMBERS)):
                dimension_number = DIMENSION_NUMBERS[d_ptr]
                for num in range(SIZES[spec_type]):
                    if dimension_number != 2 or range_ptr != 0:
                        continue

                    command = f'python {marabou_path} {onnx_dir_path}/pensieve_small_{MODEL_TYPES[spec_type]}_marabou.onnx {txt_dir_path}/pensieve_{SPEC_TYPES[spec_type]}_{dimension_number}_{range_ptr}_{num}.txt | tee {running_result_path}/pensieve_small_{SPEC_TYPES[spec_type]}_{dimension_number}_{range_ptr}_{num}.txt'

                    print("------------------------------------->")
                    print(command)
                    print("<------------------------------------->")
                    os.system(command)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: Decima_marabou_run.py marabou_path")
        exit(1)
    marabou_path = sys.argv[1]
    main(marabou_path)
