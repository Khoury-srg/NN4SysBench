import sys
import os

import random
import numpy as np




P_RANGE = [0.05, 0.1, 0.5, 0.7, 1]

MODELS = ['empty', 'small', 'mid', 'big']
DIFFICULTY = ['easy']
SIZES = [10, 10, 10]
SPEC_TYPES = [1, 2, 3]


DIMENSION_NUMBERS=[1,2,3, 4]



# responsible for writing the file
def write_vnnlib(X, spec_type, spec_path, Y_shape=6):
    with open(spec_path, "w") as f:
        f.write("\n")
        for i in range(int(X.shape[0] / 2)):
            f.write(f"(declare-const X_{i} Real)\n")
        if spec_type == SPEC_TYPES[0] or spec_type == SPEC_TYPES[1]:
            for i in range(6):
                f.write(f"(declare-const Y_{i} Real)\n")
        if spec_type == SPEC_TYPES[2]:
            f.write(f"(declare-const Y_0 Real)\n")
        f.write("\n; Input constraints:\n")

        for i in range(X.shape[0]):
            if i % 2 == 0:
                f.write(f"(assert (>= X_{int(i / 2)} {X[i]}))\n")
            else:
                f.write(f"(assert (<= X_{int((i - 1) / 2)} {X[i]}))\n")

        f.write("\n; Output constraints:\n")
        if spec_type == SPEC_TYPES[0] or spec_type == SPEC_TYPES[1]:
            if spec_type == SPEC_TYPES[0]:
                cannot_be_largest = 0
            if spec_type == SPEC_TYPES[1]:
                cannot_be_largest = Y_shape - 1
            for i in range(Y_shape):
                if not i == cannot_be_largest:
                    f.write(f"(assert (<= Y_{i} Y_{cannot_be_largest}))\n")

        if spec_type == SPEC_TYPES[2]:
            f.write(f"(assert (<= Y_0 0))\n\n")


def write_txt(X, spec_type, spec_path, Y_shape=6):
    with open(spec_path, "w") as f:
        for i in range(X.shape[0]):
            if i % 2 == 0:
                f.write(f"x{int(i / 2)} >= {X[i]}\n")
            else:
                f.write(f"x{int((i - 1) / 2)} <= {X[i]}\n")

        if spec_type == SPEC_TYPES[0] or spec_type == SPEC_TYPES[1]:
            if spec_type == SPEC_TYPES[0]:
                cannot_be_largest = 0
            if spec_type == SPEC_TYPES[1]:
                cannot_be_largest = Y_shape - 1
            for i in range(Y_shape):
                if not i == cannot_be_largest:
                    f.write(f"+y{i} -y{cannot_be_largest} <= 0\n")

        if spec_type == SPEC_TYPES[2]:
            f.write(f"y0 <= 0\n\n")


def add_range(X, spec_type, p_range,dimension_number):
    ret = np.empty(X.shape[0] * 2)
    if spec_type == SPEC_TYPES[0] or spec_type == SPEC_TYPES[1]:
        if dimension_number==1:
            for i in range(X.shape[0]):
                if 15 < i < 24:
                    ret[i * 2] = X[i]
                    ret[i * 2 + 1] = X[i] + p_range
                else:
                    ret[i * 2] = X[i]
                    ret[i * 2 + 1] = X[i]
        if dimension_number==2:
            for i in range(X.shape[0]):
                if 15 < i < 24:
                    ret[i * 2] = X[i]
                    ret[i * 2 + 1] = X[i] + p_range
                if 23 < i < 32:
                    ret[i * 2] = max(0, X[i] - p_range)
                    ret[i * 2 + 1] = X[i]
                else:
                    ret[i * 2] = X[i]
                    ret[i * 2 + 1] = X[i]
        if dimension_number==3:
            for i in range(X.shape[0]):
                if 15 < i < 24:
                    ret[i * 2] = X[i]
                    ret[i * 2 + 1] = X[i] + p_range
                if -1 < i < 8:
                    ret[i * 2] = X[i]
                    ret[i * 2 + 1] = X[i] + p_range
                if 23 < i < 32:
                    ret[i * 2] = max(0, X[i] - p_range)
                    ret[i * 2 + 1] = X[i]
                else:
                    ret[i * 2] = X[i]
                    ret[i * 2 + 1] = X[i]



    if spec_type == SPEC_TYPES[2]:
        if dimension_number==1:
            for i in range(X.shape[0]):
                if 15 < i < 24 or 63 < i < 72:
                    ret[i * 2] = X[i]
                    ret[i * 2 + 1] = X[i] + p_range
                else:
                    ret[i * 2] = X[i]
                    ret[i * 2 + 1] = X[i]
        if dimension_number==2:
            for i in range(X.shape[0]):
                if 15 < i < 24 or 63 < i < 72:
                    ret[i * 2] = X[i]
                    ret[i * 2 + 1] = X[i] + p_range
                if 23 < i < 32 or 71 < i < 80:
                    ret[i * 2] = max(0, X[i] - p_range)
                    ret[i * 2 + 1] = X[i]
                else:
                    ret[i * 2] = X[i]
                    ret[i * 2 + 1] = X[i]
        if dimension_number==3:
            for i in range(X.shape[0]):
                if 15 < i < 24 or 63 < i < 72:
                    ret[i * 2] = X[i]
                    ret[i * 2 + 1] = X[i] + p_range
                if -1 < i < 8 or 47 < i < 56:
                    ret[i * 2] = X[i]
                    ret[i * 2 + 1] = X[i] + p_range
                if 23 < i < 32 or 71 < i < 80:
                    ret[i * 2] = max(0, X[i] - p_range)
                    ret[i * 2 + 1] = X[i]
                else:
                    ret[i * 2] = X[i]
                    ret[i * 2 + 1] = X[i]
    return ret


def parser(num):
    index = int(num % 10000)
    num = int(num / 10000)

    p_range = P_RANGE[num % 10]
    num = int(num / 10)

    model = MODELS[num]
    return index, p_range, model


def get_time(all_dic, index):
    for i in range(all_dic.shape[0]):
        if (all_dic[i][0] == index):
            return all_dic[i][1], all_dic[i][2]
    return -1, -1


def gene_spec():
    vnn_dir_path = 'vnnlib'
    marabou_txt_dir_path = 'marabou_txt'
    onnx_dir_path = 'onnx'
    csv_data = []
    if not os.path.exists(marabou_txt_dir_path):
        os.makedirs(marabou_txt_dir_path)
    if not os.path.exists(vnn_dir_path):
        os.makedirs(vnn_dir_path)

    pensieve_src_path = './src/pensieve/pensieve_resources'


    for range_ptr in range(len(P_RANGE)):
        for d_ptr in range(len(DIMENSION_NUMBERS)):
            dimension_number = DIMENSION_NUMBERS[d_ptr]
            p_range = P_RANGE[range_ptr]
            for spec_type_ptr in range(len(SPEC_TYPES)):
                total_num = 0
                spec = SPEC_TYPES[spec_type_ptr]
                indexes = list(np.load(pensieve_src_path + f'/pensieve_index_{spec}.npy'))
                # dic = np.load(pensieve_src_path+f'/pen_{difficulty}.npy')

                chosen_index = random.sample(indexes, SIZES[spec_type_ptr])


                for i in chosen_index:
                    if i == 0:
                        continue
                    index, _, model = parser(i)
                    vnn_path = f'{vnn_dir_path}/pensieve_{spec}_{dimension_number}_{range_ptr}_{total_num}.vnnlib'
                    # onnx_path = onnx_dir_path + '/pensieve_' + model + '_' + spec + '.onnx'
                    input_array = np.load(pensieve_src_path + f'/pensieve_fixedInput_{spec}.npy')[index]
                    input_array_perturbed = add_range(input_array, spec, p_range,dimension_number)

                    write_vnnlib(input_array_perturbed, spec, vnn_path)
                    print(f"[Done] generate {vnn_path}")
                    txt_path = f'{marabou_txt_dir_path}/pensieve_{spec}_{dimension_number}_{range_ptr}_{total_num}.txt'
                    write_txt(input_array_perturbed, spec, txt_path)
                    print(f"[Done] generate {txt_path}")
                    total_num += 1
                    # ground_truth, timeout = get_time(dic, i)
                    # if timeout == -1:
                    #    continue
                    # csv_data.append([onnx_path, vnn_path, int(timeout)])
    return csv_data


def main(random_seed):
    random.seed(random_seed)
    return gene_spec()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: generate_properties.py <random seed>, default is 2024")
        random_seed = 2024
    else:
        random_seed = int(sys.argv[1])
    main(random_seed)
