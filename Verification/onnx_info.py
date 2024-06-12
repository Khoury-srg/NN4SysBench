import os
import sys

onnx_dir_path = '../Benchmarks/onnx'
files = os.listdir(onnx_dir_path)
onnx_info_path = "./onnx_info"

if not os.path.exists(onnx_info_path):
    os.makedirs(onnx_info_path)



for file in files:
    name = file.split('.')[0]
    out_path = f'{name}.txt'
    command = f"python -m onnx_tool -i {onnx_dir_path}/{file} -f {onnx_info_path}/{out_path}"
    print("------------------------------------->")
    print(command)
    print("<------------------------------------->")
    os.system(command)