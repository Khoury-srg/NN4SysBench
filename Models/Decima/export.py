import numpy as np
import torch.onnx
import random
import os
import onnx
import model_benchmark as model
from spark_env.env import Environment
from msg_passing_path import *
import bisect
from spark_env.job_dag import JobDAG
from spark_env.node import Node

ONNX_DIR = f'../../Benchmarks/onnx'
MODEL_LIST = ['mid']

MODEL_TYPES = ['simple','marabou']
file_path = "./best_models/model_exec50_ep_" + str(6200)
VNN_path = f'../../Benchmarks/src/decima/decima_resources'
SPEC_TYPES = [1, 2]


def load_model(actor):
    actor.load_state_dict(torch.load(file_path + "gcn.pth", map_location='cpu'), strict=False)
    actor.load_state_dict(torch.load(file_path + "gsn.pth", map_location='cpu'), strict=False)
    actor.load_state_dict(torch.load(file_path + "actor.pth", map_location='cpu'), strict=False)
    actor.eval()
    return actor


def main():
    if not os.path.exists(ONNX_DIR):
        os.makedirs(ONNX_DIR)
    for MODEL_TYPE in MODEL_TYPES:
        for MODEL in MODEL_LIST:
            for SPEC_TYPE in SPEC_TYPES:

                if MODEL_TYPE == 'simple':
                    if MODEL == 'mid':
                        save_path = ONNX_DIR + '/decima_' + MODEL + '_' + MODEL_TYPE + ".onnx"
                        actor = model.model_benchmark()
                if MODEL_TYPE == 'marabou':
                    if MODEL == 'mid':
                        save_path = ONNX_DIR + '/decima_' + MODEL + '_' + MODEL_TYPE + f"_{SPEC_TYPE}.onnx"
                        input_arrays = np.load(VNN_path + f'/decima_fixedInput_{SPEC_TYPE}_marabou.npy')
                        actor = model.model_benchmark_marabou(torch.tensor(input_arrays[0][:4300]))

                print(save_path)
                actor = load_model(actor)
                actor = actor.eval()

                if MODEL_TYPE == 'simple':
                    input = torch.zeros(1, 4300).to(torch.float32)
                if MODEL_TYPE == 'marabou':
                    input = torch.zeros(4300).to(torch.float32)

                torch_out = actor(input)
                print(torch_out)

                torch.onnx.export(actor,  # model being run
                                  input,  # model input (or a tuple for multiple inputs)
                                  save_path,  # where to save the model (can be a file or file-like object)
                                  export_params=True,  # store the trained parameter weights inside the model file
                                  opset_version=12,  # the ONNX version to export the model to
                                  do_constant_folding=True,  # whether to execute constant folding for optimization
                                  output_names=['output'])  # the model's output names

                # check the model
                actor = onnx.load(save_path)
                onnx.checker.check_model(actor)


if __name__ == '__main__':
    main()
