import sys
import os

import random
import numpy as np

P_RANGE = [1, 1.5, 2, 2.5]
MODELS = ['empty', 'small', 'mid', 'big']
DIFFICULTY = ['easy']
SIZES = [10, 10, 10]

SPEC_TYPES = [1, 2]

file_path = "../../best_models/model_exec50_ep_" + str(6200)


# responsible for writing the file
def write_vnnlib(X, cannot_be_highest, spec_type, spec_path):
    # Y1 is the number of nodes, y2 is the the model should choose before perturb
    with open(spec_path, "w") as f:
        f.write("\n")
        for i in range(int(X.shape[0] / 2)):
            f.write(f"(declare-const X_{i} Real)\n")
            f.write(f"(declare-const X_{i} Real)\n")
        for i in range(20):
            f.write(f"(declare-const Y_{i} Real)\n")
        f.write("\n; Input constraints:\n")

        for i in range(X.shape[0]):
            if i % 2 == 0:
                if X[i] > X[i + 1]:
                    print("error")
                f.write(f"(assert (>= X_{int(i / 2)} {X[i]}))\n")
            else:
                f.write(f"(assert (<= X_{int((i - 1) / 2)} {X[i]}))\n")

        f.write("\n; Output constraints:\n")
        print(spec_type)
        if spec_type == 1 or spec_type == 2:
            for i in range(20):
                if i == cannot_be_highest:
                    continue
                else:
                    f.write(f"(assert (<= Y_{i} Y_{cannot_be_highest}))\n")


def write_txt(X, cannot_be_highest, spec_type, spec_path):
    with open(spec_path, "w") as f:
        for i in range(X.shape[0]):
            if i % 2 == 0:
                f.write(f"x{int(i / 2)} >= {X[i]}\n")
            else:
                f.write(f"x{int((i - 1) / 2)} <= {X[i]}\n")

        if spec_type == 1 or spec_type == 2:
            for i in range(20):
                if i == cannot_be_highest:
                    continue
                else:
                    f.write(f"+y{i} -y{cannot_be_highest + int(i / 20) * 20} <= 0\n")
        if spec_type == 3:
            for i in range(60):
                if i % 20 == cannot_be_highest:
                    continue
                else:
                    f.write(f"+y{i} -y{cannot_be_highest + int(i / 20) * 20} <= 0\n")


def add_range(input, spec_type, p_range):
    X = input[:4300]
    ret = np.empty(X.shape[0] * 2)
    if spec_type == SPEC_TYPES[0]:
        cannot_be_highest = input[4300]
        for i in range(X.shape[0]):
            if i == cannot_be_highest * 5 + 3 or i == cannot_be_highest * 5 + 4:
                ret[i * 2] = X[i]
                ret[i * 2 + 1] = X[i] * 20*p_range if X[i] > 0 else X[i] * 0.05
            else:
                ret[i * 2] = X[i]
                ret[i * 2 + 1] = X[i]
    if spec_type == SPEC_TYPES[1]:
        cannot_be_highest = input[4300]
        child = []
        for i in range(4301, 4321):
            if input[i] == -1:
                break
            else:
                child.append(input[i])
        for i in range(X.shape[0]):
            ret[i * 2] = X[i]
            ret[i * 2 + 1] = X[i]
        for i in child:
            index = int(5 * i + 3)
            ret[index * 2 + 1] = 10 * ret[index * 2 + 1]*p_range if ret[index * 2 + 1] > 0 else ret[index * 2 + 1] * 0.05
            index = int(5 * i + 4)
            ret[index * 2 + 1] = 10 * ret[index * 2 + 1]*p_range if ret[index * 2 + 1] > 0 else ret[index * 2 + 1] * 0.05

    return ret, cannot_be_highest


def parser(num):
    print(num)
    index = int(num % 10000)
    num = int(num / 10000)

    p_range = P_RANGE[num % 10]
    num = int(num / 10)

    model = MODELS[num]
    return index, p_range, model


def gene_spec():
    if not os.path.exists('vnnlib'):
        os.makedirs('vnnlib')
    if not os.path.exists('marabou_txt'):
        os.makedirs('marabou_txt')
    vnn_dir_path = 'vnnlib'
    marabou_txt_dir_path = 'marabou_txt'

    for spec_type_ptr in range(len(SPEC_TYPES)):


        total_num = 0
        indexes = list(np.load(f'./src/decima/decima_resources/decima_index_{SPEC_TYPES[spec_type_ptr]}.npy'))
        input_arrays = np.load(f'./src/decima/decima_resources/decima_fixedInput_{SPEC_TYPES[spec_type_ptr]}.npy')
        chosen_index = random.sample(indexes, SIZES[spec_type_ptr])

        for i in chosen_index:
            if i == 0:
                continue
            index, range_ptr, model = parser(i)
            spec = SPEC_TYPES[spec_type_ptr]

            input_array = input_arrays[index]

            input_array_perturbed, cannot_be_highest = add_range(input_array, spec,P_RANGE[range_ptr])

            vnn_path = f'{vnn_dir_path}/decima_{spec}_{total_num}.vnnlib'
            write_vnnlib(input_array_perturbed, int(cannot_be_highest), spec, vnn_path)
            print(f"[Done] generate {vnn_path}")

            total_num += 1
        total_num = 0
        input_array = np.load(f'./src/decima/decima_resources/decima_fixedInput_{SPEC_TYPES[spec_type_ptr]}_marabou.npy')[0]
        spec = SPEC_TYPES[spec_type_ptr]
        input_array_perturbed, cannot_be_highest = add_range(input_array, spec, P_RANGE[0])
        txt_path = f'{marabou_txt_dir_path}/decima_{spec}_{total_num}.txt'
        write_txt(input_array_perturbed, int(cannot_be_highest), spec, txt_path)
        print(f"[Done] generate {txt_path}")


def main(random_seed):
    random.seed(random_seed)
    gene_spec()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: decima_gen.py <random seed>")
        random_seed = 2024
    else:
        random_seed = int(sys.argv[1])
    main(random_seed)
