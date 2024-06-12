import sys
import os

import random
import numpy as np

STATISTIC_RANGE = [0.005, 0.1, 0, 1]
P_RANGE = [0.8, 1, 1.2, 1.4, 1.6]
MODELS = ['empty', 'small', 'mid', 'big']
DIFFICULTY = ['easy']
SIZES = [10, 10, 10, 10, 10]

SPEC_TYPES = [101, 102, 2, 3, 4]
SPEC_ARRAY_LENGTH = [30, 30, 30, 60, 150]
SPEC_ARRAY_NUM = 3000
HISTORY = 10
DIMENSION_NUMBERS = [1, 2, 3]


# responsible for writing the file
def write_vnnlib(X, spec_type, spec_path):
    with open(spec_path, "w") as f:
        f.write("\n")
        constant_index = 0
        for i in range(int(X.shape[0] / 2)):
            f.write(f"(declare-const X_{i} Real)\n")
            constant_index = i
        constant_index += 1

        if spec_type == SPEC_TYPES[4]:
            f.write(f"(declare-const X_{constant_index} Real)\n")

        f.write(f"(declare-const Y_0 Real)\n")
        f.write("\n; Input constraints:\n")

        for i in range(X.shape[0]):
            if i % 2 == 0:
                f.write(f"(assert (>= X_{int(i / 2)} {X[i]}))\n")
            else:
                f.write(f"(assert (<= X_{int((i - 1) / 2)} {X[i]}))\n")
            constant_index = i
        constant_index += 1
        if spec_type == SPEC_TYPES[4]:
            f.write(f"(assert (>= X_{int(constant_index / 2)} 1.0))\n")
            f.write(f"(assert (<= X_{int(constant_index / 2)} 1.0))\n")

        f.write("\n; Output constraints:\n")
        if spec_type == SPEC_TYPES[0]:
            f.write(f"(assert (<= Y_0 0))\n\n")
        if spec_type == SPEC_TYPES[1]:
            f.write(f"(assert (>= Y_0 0))\n\n")
        if spec_type == SPEC_TYPES[2]:
            f.write(f"(assert (>= Y_0 0))\n\n")
        if spec_type == SPEC_TYPES[3]:
            f.write(f"(assert (<= Y_0 0))\n\n")
        if spec_type == SPEC_TYPES[4]:
            f.write(f"(assert (<= Y_0 1))\n\n")


def write_txt(X, spec_type, spec_path):
    with open(spec_path, "w") as f:
        constant_index = 0
        for i in range(X.shape[0]):
            if i % 2 == 0:
                f.write(f"x{int(i / 2)} >= {X[i]}\n")
            else:
                f.write(f"x{int((i - 1) / 2)} <= {X[i]}\n")
            constant_index = i
        constant_index += 1

        if spec_type == SPEC_TYPES[4]:
            f.write(f"x{int((constant_index) / 2)} >= 1.0\n")
            f.write(f"x{int((constant_index) / 2)} <= 1.0\n")

        if spec_type == SPEC_TYPES[0]:
            f.write(f"y0 <= 0")
        if spec_type == SPEC_TYPES[1]:
            f.write(f"y0 >= 0")
        if spec_type == SPEC_TYPES[2]:
            f.write(f"y0 >= 0")
        if spec_type == SPEC_TYPES[3]:
            f.write(f"y0 <= 0")
        if spec_type == SPEC_TYPES[4]:
            f.write(f"y0 <= 0")


def add_range(X, spec_type, p_range, dimension):
    ret = np.empty(X.shape[0] * 2)
    if spec_type == SPEC_TYPES[0]:
        for i in range(X.shape[0]):
            if i < 10:
                ret[i * 2] = X[i]
                ret[i * 2 + 1] = X[i] + STATISTIC_RANGE[0] * p_range

            if 9 < i < 20:
                ret[i * 2] = X[i]
                ret[i * 2 + 1] = X[i] + STATISTIC_RANGE[0] * p_range
            if 19 < i < 30:
                ret[i * 2] = X[i]
                ret[i * 2 + 1] = X[i] + STATISTIC_RANGE[2] * p_range
    if spec_type == SPEC_TYPES[1]:
        for i in range(X.shape[0]):
            if dimension == 1:
                if i < 10:
                    ret[i * 2] = X[i]
                    ret[i * 2 + 1] = X[i] + STATISTIC_RANGE[1] * p_range
                else:
                    ret[i * 2] = X[i]
                    ret[i * 2 + 1] = X[i]
            if dimension == 2:
                if i < 10:
                    ret[i * 2] = X[i]
                    ret[i * 2 + 1] = X[i] + STATISTIC_RANGE[1] * p_range
                if 9 < i < 20:
                    ret[i * 2] = X[i]
                    ret[i * 2 + 1] = X[i] + STATISTIC_RANGE[3] * p_range
                else:
                    ret[i * 2] = X[i]
                    ret[i * 2 + 1] = X[i]
            if dimension == 3:

                if i < 10:
                    ret[i * 2] = X[i]
                    ret[i * 2 + 1] = X[i] + STATISTIC_RANGE[1] * p_range
                if 9 < i < 20:
                    ret[i * 2] = X[i]
                    ret[i * 2 + 1] = X[i] + STATISTIC_RANGE[3] * p_range
                if 19 < i < 30:
                    ret[i * 2] = X[i]
                    ret[i * 2 + 1] = X[i] + STATISTIC_RANGE[3] * p_range
    if spec_type == SPEC_TYPES[2]:
        for i in range(X.shape[0]):
            ret[i * 2] = X[i]
            ret[i * 2 + 1] = X[i]
            if 19 < i < 30:
                ret[i * 2 + 1] = X[i] + STATISTIC_RANGE[3] * p_range
    if spec_type == SPEC_TYPES[3]:
        for i in range(X.shape[0]):
            ret[i * 2] = X[i]
            ret[i * 2 + 1] = X[i]
            j = i % 30
            if j < 10:
                ret[i * 2 + 1] = X[i] + STATISTIC_RANGE[0] * p_range
            if 9 < j < 20:
                ret[i * 2 + 1] = X[i] + STATISTIC_RANGE[0] * p_range
    if spec_type == SPEC_TYPES[4]:
        for i in range(X.shape[0]):
            ret[i * 2] = X[i]
            ret[i * 2 + 1] = X[i]
            j = i % 30
            if j < 10:
                ret[i * 2 + 1] = X[i] + STATISTIC_RANGE[0] * p_range
            if 9 < j < 20:
                ret[i * 2 + 1] = X[i] + STATISTIC_RANGE[0] * p_range
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
    aurora_src_path = './src/aurora/aurora_resources'
    vnn_dir_path = 'vnnlib'
    onnx_dir_path = 'onnx'
    marabou_txt_dir_path = 'marabou_txt'

    if not os.path.exists(marabou_txt_dir_path):
        os.makedirs(marabou_txt_dir_path)
    if not os.path.exists(vnn_dir_path):
        os.makedirs(vnn_dir_path)

    for range_ptr in range(len(P_RANGE)):
        for d_ptr in range(len(DIMENSION_NUMBERS)):
            dimension_number = DIMENSION_NUMBERS[d_ptr]
            p_range = P_RANGE[range_ptr]

            for spec_type_ptr in range(len(SPEC_TYPES)):
                total_num = 0
                indexes = list(np.load(aurora_src_path + f'/aurora_index_{SPEC_TYPES[spec_type_ptr]}.npy'))
                # dic = np.load(f'./src/pensieve/pensieve_resources/pen_{difficulty}_dic.npy')
                chosen_index = random.sample(indexes, SIZES[spec_type_ptr])

                for i in chosen_index:
                    if i == 0:
                        continue
                    if spec_type_ptr != 1 and dimension_number != 3 and p_range != 1:
                        continue
                    index, _, model = parser(i)
                    spec = SPEC_TYPES[spec_type_ptr]
                    vnn_path = f'{vnn_dir_path}/aurora_{spec}_{dimension_number}_{range_ptr}_{total_num}.vnnlib'
                    onnx_path = onnx_dir_path + '/pensieve_' + model + '_' + str(spec) + '.onnx'
                    input_array = np.load(aurora_src_path + f'/aurora_fixedInput_{SPEC_TYPES[spec_type_ptr]}.npy')[
                        index]
                    input_array_perturbed = add_range(input_array, spec, p_range, dimension_number)
                    write_vnnlib(input_array_perturbed, spec, vnn_path)
                    print(f"[Done] generate {vnn_path}")
                    txt_path = f'{marabou_txt_dir_path}/aurora_{spec}_{dimension_number}_{range_ptr}_{total_num}.txt'
                    write_txt(input_array_perturbed, spec, txt_path)
                    print(f"[Done] generate {txt_path}")
                    total_num += 1
                    # ground_truth, timeout = get_time(dic, i)
                    # if timeout == -1:
                    # continue
                    # csv_data.append([onnx_path, vnn_path, int(timeout)])
        # return csv_data


def main(random_seed):
    random.seed(random_seed)
    gene_spec()
    # return gene_spec()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: aurora_gen.py <random seed>")
        random_seed = 2024
    else:
        random_seed = int(sys.argv[1])
    main(random_seed)
