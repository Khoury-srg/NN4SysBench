'''
This script it to generate the instances for Pensieve.
'''

import os
import random
import argparse
import numpy as np

# Global parameters
P_RANGE = [0.05, 0.1, 0.5, 0.7, 1]  # Perturbation range for inputs
MODELS = ['empty', 'small', 'mid', 'big']  # Available models
SPEC_TYPES = [1, 2, 3]  # Different specification types
DIMENSION_NUMBERS = [1, 2, 3, 4]  # Dimension numbers for perturbation
# DIFFICULTY = ['easy'] # we will finetune difficulties in the future.
DEFAULT_SIZES = [10, 10, 10]


# Function to write the VNNLIB format file
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
            cannot_be_largest = 0 if spec_type == SPEC_TYPES[0] else Y_shape - 1
            for i in range(Y_shape):
                if i != cannot_be_largest:
                    f.write(f"(assert (<= Y_{i} Y_{cannot_be_largest}))\n")
        if spec_type == SPEC_TYPES[2]:
            f.write(f"(assert (<= Y_0 0))\n\n")


# Function to write the TXT format file
def write_txt(X, spec_type, spec_path, Y_shape=6):
    with open(spec_path, "w") as f:
        for i in range(X.shape[0]):
            if i % 2 == 0:
                f.write(f"x{int(i / 2)} >= {X[i]}\n")
            else:
                f.write(f"x{int((i - 1) / 2)} <= {X[i]}\n")

        if spec_type == SPEC_TYPES[0] or spec_type == SPEC_TYPES[1]:
            cannot_be_largest = 0 if spec_type == SPEC_TYPES[0] else Y_shape - 1
            for i in range(Y_shape):
                if i != cannot_be_largest:
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

# Function to parse the selected number into property
# eg: a number ABCDEF, CDEF is the instance index, B is the perturbation range index, A is the model index
def my_parser(num):
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


def gene_spec(sizes=DEFAULT_SIZES):
    vnn_dir_path = 'vnnlib'
    marabou_txt_dir_path = 'marabou_txt'
    onnx_dir_path = 'onnx'
    csv_data = []

    # Ensure required directories exist
    for dir_path in [vnn_dir_path, marabou_txt_dir_path, onnx_dir_path]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    pensieve_src_path = './src/pensieve/pensieve_resources'

    # Iterate over perturbation ranges
    for range_ptr in range(len(P_RANGE)):
        p_range = P_RANGE[range_ptr]
        
        total = 0

        # Iterate over dimension numbers
        for d_ptr in range(len(DIMENSION_NUMBERS)):
            dimension_number = DIMENSION_NUMBERS[d_ptr]

            # Iterate over specification types
            for spec_type_ptr in range(len(SPEC_TYPES)):
                spec = SPEC_TYPES[spec_type_ptr]
                total_num = 0
                
                # Load index for current specification type
                indexes = list(np.load(pensieve_src_path + f'/pensieve_index_{spec}.npy'))
                
                # Randomly select indexes based on size for this spec type
                chosen_index = random.sample(indexes, sizes[spec_type_ptr])

                # TODO: This dictionary is for VNN competition. We will update it ASAP we have the standard timeout 
                # dic = np.load(pensieve_src_path+f'/pen_{difficulty}.npy')

                # Iterate through the chosen indexes
                for i in chosen_index:
                    if i == 0:
                        continue
                    
                    # Parse index and retrieve model information
                    index, _, _ = my_parser(i)

                    # Define paths for VNNLIB and TXT files
                    vnn_path = f'{vnn_dir_path}/pensieve_{spec}_{dimension_number}_{range_ptr}_{total_num}.vnnlib'
                    txt_path = f'{marabou_txt_dir_path}/pensieve_{spec}_{dimension_number}_{range_ptr}_{total_num}.txt'
                    
                    # Load input array and generate perturbed inputs
                    input_array = np.load(pensieve_src_path + f'/pensieve_fixedInput_{spec}.npy')[index]
                    input_array_perturbed = add_range(input_array, spec, p_range, dimension_number)

                    # Write the VNNLIB and TXT files
                    write_vnnlib(input_array_perturbed, spec, vnn_path)
                    print(f"[Done] generate {vnn_path}")

                    write_txt(input_array_perturbed, spec, txt_path)
                    print(f"[Done] generate {txt_path}")

                    total_num += 1
                    total += 1

                    # If timeout values are available, append the data to csv_data
                    # TODO: The CSV data is for VNN competition, current we do not provide standard timeout for the benchmark.
                    # ground_truth, timeout = get_time(dic, i)
                    # if timeout == -1:
                    #     continue
                    # csv_data.append([onnx_path, vnn_path, int(timeout)])

    print(f"[Done] Successfully generate {total} instances for Pensieve.")
    return csv_data


# Main function to run the specification generation
def main(random_seed, sizes=DEFAULT_SIZES):
    """
    Main entry point of the script. Sets the random seed and triggers the generation of specifications.
    :param random_seed: The seed for random number generation
    :param sizes: A list of sizes for each specification type
    """
    random.seed(random_seed)
    return gene_spec(sizes)


# Entry point for command-line execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2024, help="Random seed for reproducibility")
    parser.add_argument("--sizes", type=int, nargs=3, default=[10, 10, 10], help="Sizes for each specification type")
    args = parser.parse_args()

    random_seed = args.seed
    sizes = args.sizes

    # Execute the main function with the provided seed and sizes
    main(random_seed, sizes)
