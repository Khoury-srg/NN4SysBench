import numpy as np
import random
import os

from data_generator_benchmark import get_inputs_array

MODEL_LIST = ['small', 'mid', 'big']
SPEC_TYPES = [1, 2, 3]

SIZE = 3000
RANDOMSEED = 2024
DIR = f'../../Benchmarks/src/pensieve/pensieve_resources'


def gene_spec():
    for spec_type in SPEC_TYPES:
        if spec_type == SPEC_TYPES[0] or spec_type == SPEC_TYPES[1]:
            myarr = np.empty((SIZE, 48))
        if spec_type == SPEC_TYPES[2]:
            myarr = np.empty((SIZE, 96))
        for i in range(SIZE):
            if spec_type == SPEC_TYPES[0]:
                    # specification 1
                X = get_inputs_array(1, random_seed=i).flatten()
            elif spec_type == SPEC_TYPES[1]:
                    # specification 2
                X = get_inputs_array(2, random_seed=i).flatten()
            else:
                # specification 3
                X = get_inputs_array(3, random_seed=i).flatten()
            myarr[i] = X
        np.save(DIR + f'/pensieve_fixedInput_{spec_type}.npy', myarr)


def gen_index():
    for i in range(0, len(SPEC_TYPES)):
        index_arr = np.empty(SIZE)
        for j in range(SIZE):
            # first number is model number, second is range number
            index_arr[j] = 200000 + j

        np.save(DIR + f'/pensieve_index_{SPEC_TYPES[i]}.npy', index_arr)


def main():
    random.seed(RANDOMSEED)
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    gene_spec()
    gen_index()


if __name__ == "__main__":
    main()
