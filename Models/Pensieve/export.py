import numpy as np
import torch.onnx
import random
import onnx
import model_no_softmax_no_argmax as model
import data_generator_benchmark as data_generator

S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
SUMMARY_DIR = './test_results'
ONNX_DIR = f'../../Benchmarks/onnx'
NN_MODELS = ['./results/nn_model_ep_BEST_small_linear_10600.pth', './results/nn_model_ep_BEST_mid_linear_7200.pth',
             './results/nn_model_ep_BEST_big_linear_10900.pth']
MODEL_LIST = ['small', 'mid', 'big']
MODEL_TYPES = ['simple', 'parallel']
VERIFIERS = ['abcrown', 'marabou']


def create_bw(condition="good"):
    bw = torch.zeros(1, 8)
    for i in range(8):
        bw[0, i] = 10 + 10 * random.random()
    return bw


def create_video_size(smallest_size=7400):
    video_size = torch.zeros(1, 8)
    video_size[0, 0] = smallest_size + random.random() * 100
    return video_size


def main():
    for verifier in VERIFIERS:
        if verifier == 'abcrown':
            for MODEL_ptr in range(len(MODEL_LIST)):
                for MODEL_TYPE in MODEL_TYPES:
                    MODEL = MODEL_LIST[MODEL_ptr]
                    NN_MODEL = NN_MODELS[MODEL_ptr]

                    save_path = ONNX_DIR + '/pensieve_' + MODEL + '_' + MODEL_TYPE + ".onnx"
                    print(save_path)
                    if MODEL_TYPE == 'simple':
                        if MODEL == 'mid':
                            actor = model.ActorNetwork_mid(state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                                           learning_rate=ACTOR_LR_RATE)
                        if MODEL == 'big':
                            actor = model.ActorNetwork_big(state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                                           learning_rate=ACTOR_LR_RATE)
                        if MODEL == 'small':
                            actor = model.ActorNetwork_small(state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                                             learning_rate=ACTOR_LR_RATE)

                        # run one time to test
                        myinput = torch.zeros(1, 6, 8)  # Define your input here
                        torch_out = actor(myinput)
                        print(torch_out)

                        # export
                        para = torch.load(NN_MODEL, map_location=torch.device('cpu'))
                        actor.load_state_dict(para)
                        actor = actor.eval()

                        torch.onnx.export(actor,  # model being run
                                          myinput,  # model input (or a tuple for multiple inputs)
                                          save_path,  # where to save the model (can be a file or file-like object)
                                          export_params=True,
                                          # store the trained parameter weights inside the model file
                                          opset_version=12,  # the ONNX version to export the model to
                                          do_constant_folding=True,
                                          # whether to execute constant folding for optimization
                                          input_names=['input', 'bw', 'video_size'],  # the model's input names
                                          output_names=['output'])  # the model's output names

                    if MODEL_TYPE == 'parallel':
                        if MODEL == 'mid':
                            actor = model.ActorNetwork_mid_parallel(state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                                                    learning_rate=ACTOR_LR_RATE)
                        if MODEL == 'big':
                            actor = model.ActorNetwork_big_parallel(state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                                                    learning_rate=ACTOR_LR_RATE)
                        if MODEL == 'small':
                            actor = model.ActorNetwork_small_parallel(state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                                                      learning_rate=ACTOR_LR_RATE)

                        # export
                        para = torch.load(NN_MODEL, map_location=torch.device('cpu'))
                        actor.load_state_dict(para)

                        # run one time to test
                        actor = actor.eval()
                        myinput = data_generator.get_inputs_array(3, random_seed=1)
                        myinput = torch.from_numpy(myinput).to(torch.float32)
                        # myinput2 = data_generator.get_inputs_array(2, random_seed=4)
                        # myinput2 = torch.from_numpy(myinput2).to(torch.float32)
                        # myinput = torch.concat([myinput, myinput2])
                        # myinput = myinput.view(2, 12, 8)
                        print(myinput)
                        torch_out = actor(myinput)
                        print("-----")
                        print(torch_out)

                        torch.onnx.export(actor,  # model being run
                                          myinput,  # model input (or a tuple for multiple inputs)
                                          save_path,  # where to save the model (can be a file or file-like object)
                                          export_params=True,
                                          # store the trained parameter weights inside the model file
                                          opset_version=12,  # the ONNX version to export the model to
                                          do_constant_folding=False,
                                          # whether to execute constant folding for optimization
                                          input_names=['input', 'bw', 'video_size'],  # the model's input names
                                          output_names=['output'])  # the model's output names

                    # check the model
                    actor = onnx.load(save_path)
                    onnx.checker.check_model(actor)
        if verifier == 'marabou':
            for MODEL_ptr in [0]:
                for MODEL_TYPE in MODEL_TYPES:
                    MODEL = MODEL_LIST[MODEL_ptr]
                    NN_MODEL = NN_MODELS[MODEL_ptr]

                    save_path = ONNX_DIR + '/pensieve_' + MODEL + '_' + MODEL_TYPE + "_marabou.onnx"
                    print(save_path)
                    if MODEL_TYPE == 'simple':
                        if MODEL == 'mid':
                            actor = model.ActorNetwork_mid_marabou(state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                                                   learning_rate=ACTOR_LR_RATE)
                        if MODEL == 'big':
                            actor = model.ActorNetwork_big_marabou(state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                                                   learning_rate=ACTOR_LR_RATE)
                        if MODEL == 'small':
                            actor = model.ActorNetwork_small_marabou(state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                                                     learning_rate=ACTOR_LR_RATE)

                        # run one time to test
                        myinput = torch.zeros(6, 8)  # Define your input here
                        torch_out = actor(myinput)
                        print(torch_out)

                        # export
                        para = torch.load(NN_MODEL, map_location=torch.device('cpu'))
                        actor.load_state_dict(para)
                        actor = actor.eval()

                        torch.onnx.export(actor,  # model being run
                                          myinput,  # model input (or a tuple for multiple inputs)
                                          save_path,  # where to save the model (can be a file or file-like object)
                                          export_params=True,
                                          # store the trained parameter weights inside the model file
                                          opset_version=12,  # the ONNX version to export the model to
                                          do_constant_folding=True,
                                          # whether to execute constant folding for optimization
                                          input_names=['input', 'bw', 'video_size'],  # the model's input names
                                          output_names=['output'])  # the model's output names

                    if MODEL_TYPE == 'parallel':
                        continue

                    # check the model
                    actor = onnx.load(save_path)
                    onnx.checker.check_model(actor)


if __name__ == '__main__':
    main()
