import os
import torch
import onnx

ONNX_DIR = f'../../Benchmarks/onnx'





def main():
    if not os.path.exists(ONNX_DIR):
        os.makedirs(ONNX_DIR)
    models = ["mscn_128d.onnx", "mscn_128d_dual.onnx", "mscn_2048d.onnx", "mscn_2048d_dual.onnx"]
    # copy_name = ["cardinality_128.onnx", "cardinality_128_dual.onnx", "cardinality_2048.onnx", "cardinality_2048_dual.onnx"]
    for i in range(len(models)):
        os.system(f'cp {models[i]} {ONNX_DIR}/{models[i]}')



if __name__ == "__main__":
    main()
