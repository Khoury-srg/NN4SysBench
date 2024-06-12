import os
import sys

MODELS = ['mid', 'mid', 'mid']
MODEL_TYPES = ['simple', 'simple']
SIZES = [10, 10, 10]

SPEC_TYPES = [1, 2]

running_result_path = './decima_marabou_running_result'

txt_dir_path = '../Benchmarks/marabou_txt'
onnx_dir_path = '../Benchmarks/onnx'

if not os.path.exists(running_result_path):
    os.makedirs(running_result_path)

def main(marabou_path):
    for spec_type_ptr in range(len(SPEC_TYPES)):
        for i in range(10):
            command = f'python {marabou_path} {onnx_dir_path}/decima_mid_marabou_{SPEC_TYPES[spec_type_ptr]}.onnx {txt_dir_path}/decima_{SPEC_TYPES[spec_type_ptr]}_0.txt | tee {running_result_path}/decima_mid_{SPEC_TYPES[spec_type_ptr]}_{i}.txt'
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
