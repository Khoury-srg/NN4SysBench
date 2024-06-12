import os
import sys



running_result_path = './lindex_marabou_running_result'

SIZE = 10
MODEL_NAMES=["lindex","lindex_deep"]



txt_dir_path = '../Benchmarks/marabou_txt'
onnx_dir_path = '../Benchmarks/onnx'


def main(marabou_path):
    if not os.path.exists(running_result_path):
        os.makedirs(running_result_path)
    model_name = "lindex"

    for model in MODEL_NAMES:
        for num in range(SIZE):
            command = f'python {marabou_path} {onnx_dir_path}/{model}.onnx {txt_dir_path}/{model_name}_0_{num}.txt | tee {running_result_path}/{model}_0_{num}.txt'
            print(command)
            os.system(command)



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: Lindex_marabou_run.py marabou_path")
        exit(1)
    marabou_path = sys.argv[1]
    main(marabou_path)
