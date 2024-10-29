'''
This script generates instances for Aurora models and creates the corresponding VNNLIB and Marabou TXT files.
'''

import sys
import os
import random
import numpy as np

# Global constants for the perturbation and model parameters
STATISTIC_RANGE = [0.005, 0.1, 0, 1]
P_RANGE = [0.8, 1, 1.2, 1.4, 1.6]
MODELS = ['empty', 'small', 'mid', 'big']
SIZES = [10, 10, 10, 10, 10]  # Default sizes if not provided
SPEC_TYPES = [101, 102, 2, 3, 4]
DIMENSION_NUMBERS = [1, 2, 3]

# Paths for storing results
AURORA_SRC_PATH = './src/aurora/aurora_resources'
VNNLIB_DIR = 'vnnlib'
ONNX_DIR = 'onnx'
MARABOU_TXT_DIR = 'marabou_txt'

# Ensure necessary directories exist
os.makedirs(VNNLIB_DIR, exist_ok=True)
os.makedirs(MARABOU_TXT_DIR, exist_ok=True)

# Function to write VNNLIB file
def write_vnnlib(X, spec_type, spec_path):
    """
    Writes the VNNLIB file format for Aurora instances.

    Args:
        X (array): The perturbed input data.
        spec_type (int): The specification type.
        spec_path (str): The path to save the generated VNNLIB file.
    """
    with open(spec_path, "w") as f:
        f.write("\n")
        for i in range(int(X.shape[0] / 2)):
            f.write(f"(declare-const X_{i} Real)\n")
        if spec_type == SPEC_TYPES[4]:
            f.write(f"(declare-const X_{int(X.shape[0] / 2)} Real)\n")
        f.write(f"(declare-const Y_0 Real)\n")

        f.write("\n; Input constraints:\n")
        for i in range(X.shape[0]):
            if i % 2 == 0:
                f.write(f"(assert (>= X_{int(i / 2)} {X[i]}))\n")
            else:
                f.write(f"(assert (<= X_{int((i - 1) / 2)} {X[i]}))\n")
        
        if spec_type == SPEC_TYPES[4]:
            f.write(f"(assert (>= X_{int(X.shape[0] / 2)} 1.0))\n")
            f.write(f"(assert (<= X_{int(X.shape[0] / 2)} 1.0))\n")

        f.write("\n; Output constraints:\n")
        if spec_type in [SPEC_TYPES[0], SPEC_TYPES[3], SPEC_TYPES[4]]:
            f.write(f"(assert (<= Y_0 0))\n")
        else:
            f.write(f"(assert (>= Y_0 0))\n")
        print(f"[DONE] Generated VNNLIB at {spec_path}")

# Function to write Marabou TXT file
def write_txt(X, spec_type, spec_path):
    """
    Writes the Marabou TXT file format for Aurora instances.

    Args:
        X (array): The perturbed input data.
        spec_type (int): The specification type.
        spec_path (str): The path to save the generated TXT file.
    """
    with open(spec_path, "w") as f:
        for i in range(X.shape[0]):
            if i % 2 == 0:
                f.write(f"x{int(i / 2)} >= {X[i]}\n")
            else:
                f.write(f"x{int((i - 1) / 2)} <= {X[i]}\n")
        
        if spec_type == SPEC_TYPES[4]:
            f.write(f"x{int(X.shape[0] / 2)} >= 1.0\n")
            f.write(f"x{int(X.shape[0] / 2)} <= 1.0\n")

        if spec_type in [SPEC_TYPES[0], SPEC_TYPES[3], SPEC_TYPES[4]]:
            f.write(f"y0 <= 0\n")
        else:
            f.write(f"y0 >= 0\n")
        print(f"[DONE] Generated TXT at {spec_path}")

# Function to perturb the input data based on the specification type and range
def add_range(X, spec_type, p_range, dimension):
    """
    Adds perturbations to the input data X based on the specification type and perturbation range.

    Args:
        X (array): The input data.
        spec_type (int): The specification type.
        p_range (float): The perturbation range.
        dimension (int): The dimension number.
    
    Returns:
        array: Perturbed data.
    """
    ret = np.empty(X.shape[0] * 2)
    if spec_type == SPEC_TYPES[0]:
        for i in range(X.shape[0]):
            ret[i * 2] = X[i]
            ret[i * 2 + 1] = X[i] + STATISTIC_RANGE[0] * p_range
    elif spec_type == SPEC_TYPES[1]:
        for i in range(X.shape[0]):
            ret[i * 2] = X[i]
            if dimension == 1:
                ret[i * 2 + 1] = X[i] + STATISTIC_RANGE[1] * p_range
            elif dimension == 2:
                ret[i * 2 + 1] = X[i] + (STATISTIC_RANGE[1] if i < 10 else STATISTIC_RANGE[3]) * p_range
            else:
                ret[i * 2 + 1] = X[i] + (STATISTIC_RANGE[1] if i < 10 else STATISTIC_RANGE[3]) * p_range
    else:
        for i in range(X.shape[0]):
            ret[i * 2] = X[i]
            ret[i * 2 + 1] = X[i]
    return ret

# Function to parse the encoded number into index, range, and model
def parser(num):
    """
    Parses the encoded number into index, perturbation range, and model type.

    Args:
        num (int): Encoded number representing index, perturbation range, and model.
    
    Returns:
        tuple: index, perturbation range, and model type.
    """
    num = int(num)
    index = num % 10000
    num //= 10000
    p_range = P_RANGE[num % 10]
    num //= 10
    model = MODELS[num]
    return index, p_range, model

# Main function to generate the specifications
def gene_spec(size):
    """
    Generates the VNNLIB and TXT specification files for Aurora instances.

    Args:
        size (int): Number of instances to generate for each specification.
    """
    for range_ptr, p_range in enumerate(P_RANGE):
        for d_ptr, dimension_number in enumerate(DIMENSION_NUMBERS):
            for spec_type in SPEC_TYPES:
                total_num = 0
                indexes = list(np.load(f'{AURORA_SRC_PATH}/aurora_index_{spec_type}.npy'))
                chosen_indexes = random.sample(indexes, size)  # Use size to determine number of instances

                for i in chosen_indexes:
                    if i == 0:
                        continue
                    index, _, model = parser(i)
                    input_array = np.load(f'{AURORA_SRC_PATH}/aurora_fixedInput_{spec_type}.npy')[index]
                    perturbed_input = add_range(input_array, spec_type, p_range, dimension_number)

                    vnnlib_path = f'{VNNLIB_DIR}/aurora_{spec_type}_{dimension_number}_{range_ptr}_{total_num}.vnnlib'
                    write_vnnlib(perturbed_input, spec_type, vnnlib_path)

                    txt_path = f'{MARABOU_TXT_DIR}/aurora_{spec_type}_{dimension_number}_{range_ptr}_{total_num}.txt'
                    write_txt(perturbed_input, spec_type, txt_path)

                    total_num += 1

# Main function to set random seed and start generation
def main(random_seed, size=10):
    """
    Main entry point of the script.

    Args:
        random_seed (int): The seed for random number generation.
        size (int): The number of instances to generate for each specification.
    """
    random.seed(random_seed)
    gene_spec(size)

if __name__ == "__main__":
    if len(sys.argv) not in [2, 3]:
        print("Usage: aurora_gen.py <random seed> [size]")
        random_seed = 2024  # Default random seed
        size = 10  # Default size
    else:
        random_seed = int(sys.argv[1])
        size = int(sys.argv[2]) if len(sys.argv) == 3 else 10

    main(random_seed, size)
