'''
This script generates instances for Decima models and creates the corresponding VNNLIB and Marabou TXT files.
'''

import sys
import os
import random
import numpy as np

# Global parameters
P_RANGE = [1, 1.5, 2, 2.5]  # Perturbation ranges for inputs
MODELS = ['empty', 'small', 'mid', 'big']  # Model sizes
SPEC_TYPES = [1, 2]  # Specification types

# Paths to input and output directories
VNNLIB_DIR = 'vnnlib'
MARABOU_TXT_DIR = 'marabou_txt'

# Ensure output directories exist
if not os.path.exists(VNNLIB_DIR):
    os.makedirs(VNNLIB_DIR)
if not os.path.exists(MARABOU_TXT_DIR):
    os.makedirs(MARABOU_TXT_DIR)


def write_vnnlib(X, cannot_be_highest, spec_type, spec_path):
    """
    Writes the VNNLIB file for verification purposes.
    
    Args:
        X (array): Perturbed input data.
        cannot_be_highest (int): Index that cannot be the highest.
        spec_type (int): Type of specification.
        spec_path (str): Path to save the generated VNNLIB file.
    """
    with open(spec_path, "w") as f:
        f.write("\n")
        for i in range(int(X.shape[0] / 2)):
            f.write(f"(declare-const X_{i} Real)\n")
        for i in range(20):
            f.write(f"(declare-const Y_{i} Real)\n")

        f.write("\n; Input constraints:\n")
        for i in range(X.shape[0]):
            if i % 2 == 0:
                f.write(f"(assert (>= X_{int(i / 2)} {X[i]}))\n")
            else:
                f.write(f"(assert (<= X_{int((i - 1) / 2)} {X[i]}))\n")

        f.write("\n; Output constraints:\n")
        if spec_type in [1, 2]:
            for i in range(20):
                if i != cannot_be_highest:
                    f.write(f"(assert (<= Y_{i} Y_{cannot_be_highest}))\n")

    print(f"[Done] Generated VNNLIB file: {spec_path}")


def write_txt(X, cannot_be_highest, spec_type, spec_path):
    """
    Writes the TXT file for Marabou verification purposes.
    
    Args:
        X (array): Perturbed input data.
        cannot_be_highest (int): Index that cannot be the highest.
        spec_type (int): Type of specification.
        spec_path (str): Path to save the generated TXT file.
    """
    with open(spec_path, "w") as f:
        for i in range(X.shape[0]):
            if i % 2 == 0:
                f.write(f"x{int(i / 2)} >= {X[i]}\n")
            else:
                f.write(f"x{int((i - 1) / 2)} <= {X[i]}\n")

        if spec_type in [1, 2]:
            for i in range(20):
                if i != cannot_be_highest:
                    f.write(f"+y{i} -y{cannot_be_highest} <= 0\n")

    print(f"[Done] Generated TXT file: {spec_path}")


def add_range(input_array, spec_type, p_range):
    """
    Adds perturbation to the input data based on the specification type and perturbation range.
    
    Args:
        input_array (array): Original input data.
        spec_type (int): Specification type.
        p_range (float): Perturbation range.
    
    Returns:
        tuple: Perturbed input data and index that cannot be the highest.
    """
    X = input_array[:4300]
    ret = np.empty(X.shape[0] * 2)

    cannot_be_highest = input_array[4300]

    if spec_type == 1:
        for i in range(X.shape[0]):
            if i == cannot_be_highest * 5 + 3 or i == cannot_be_highest * 5 + 4:
                ret[i * 2] = X[i]
                ret[i * 2 + 1] = X[i] * 20 * p_range if X[i] > 0 else X[i] * 0.05
            else:
                ret[i * 2] = X[i]
                ret[i * 2 + 1] = X[i]
    elif spec_type == 2:
        child_indices = [input_array[i] for i in range(4301, 4321) if input_array[i] != -1]
        for i in range(X.shape[0]):
            ret[i * 2] = X[i]
            ret[i * 2 + 1] = X[i]

        for i in child_indices:
            index = int(5 * i + 3)
            ret[index * 2 + 1] = 10 * ret[index * 2 + 1] * p_range if ret[index * 2 + 1] > 0 else ret[index * 2 + 1] * 0.05
            index = int(5 * i + 4)
            ret[index * 2 + 1] = 10 * ret[index * 2 + 1] * p_range if ret[index * 2 + 1] > 0 else ret[index * 2 + 1] * 0.05

    return ret, cannot_be_highest


def parse_index(num):
    """
    Parses the given number to extract the index, perturbation range, and model.
    
    Args:
        num (int): The encoded number.
    
    Returns:
        tuple: Parsed index, perturbation range, and model.
    """
    index = int(num % 10000)
    num = int(num / 10000)
    p_range = P_RANGE[num % 10]
    num = int(num / 10)
    model = MODELS[num]
    return index, p_range, model


def gene_spec(size):
    """
    Generates VNNLIB and TXT specifications for Decima models.
    
    Args:
        size (int): Number of instances to generate.
    """
    for spec_type in SPEC_TYPES:
        total_num = 0
        indexes = list(np.load(f'./src/decima/decima_resources/decima_index_{spec_type}.npy'))
        input_arrays = np.load(f'./src/decima/decima_resources/decima_fixedInput_{spec_type}.npy')
        chosen_indexes = random.sample(indexes, size)

        for i in chosen_indexes:
            if i == 0:
                continue

            index, range_ptr, model = parse_index(i)
            input_array = input_arrays[index]

            perturbed_input, cannot_be_highest = add_range(input_array, spec_type, P_RANGE[range_ptr])

            # Write VNNLIB file
            vnnlib_path = f'{VNNLIB_DIR}/decima_{spec_type}_{total_num}.vnnlib'
            write_vnnlib(perturbed_input, int(cannot_be_highest), spec_type, vnnlib_path)

            total_num += 1

        # Generate Marabou TXT file
        input_array = np.load(f'./src/decima/decima_resources/decima_fixedInput_{spec_type}_marabou.npy')[0]
        perturbed_input, cannot_be_highest = add_range(input_array, spec_type, P_RANGE[0])

        txt_path = f'{MARABOU_TXT_DIR}/decima_{spec_type}_{total_num}.txt'
        write_txt(perturbed_input, int(cannot_be_highest), spec_type, txt_path)


def main(random_seed, size=10):
    """
    Main function to generate specifications with a specified random seed and number of instances.
    
    Args:
        random_seed (int): Seed for random number generation.
        size (int): Number of instances to generate.
    """
    random.seed(random_seed)
    gene_spec(size)


if __name__ == "__main__":
    if len(sys.argv) not in [2, 3]:
        print("Usage: decima_gen.py <random seed> [size]")
        random_seed = 2024  # Default random seed
        size = 10  # Default size
    else:
        random_seed = int(sys.argv[1])
        size = int(sys.argv[2]) if len(sys.argv) == 3 else 10

    main(random_seed, size)
