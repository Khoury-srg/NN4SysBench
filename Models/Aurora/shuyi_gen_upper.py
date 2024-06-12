import os.path
import random

import numpy as np

# MODEL_LIST = ['small', 'mid', 'big']
SPEC_TYPES = [101, 102, 2, 3, 4]
SPEC_ARRAY_LENGTH = [30, 30, 30, 60, 150]
SPEC_ARRAY_NUM = 10000
HISTORY = 10
RANDOMSEED = 2024
DIR = f'../../Benchmarks/src/aurora/aurora_resources'

'''
1.1. When observe good network condition (close-to-minimum latency, no packet loss), should increase sending rate
	== when in low latency (latency gradient < 0.01, latency ratio<1.01), no packet loss (sending ratio==1), output should be positive
1.2. When observe bad network condition (high latency, observe packet loss), should decrease sending rate
	== when in low latency (latency gradient >0.2, latency ratio>2), no packet loss (sending ratio >=2), output should be negative
2. When the congestion controller is sending on a link with a shallow buffer (close-to-minimum latency) and experience high packet loss, should decrease sending rate
	== when in low latency (latency gradient < 0.01, latency ratio<1.01), packet loss (sending ratio>=2), output should be negative


3. Consistency && multiple steps: better condition should increase more sending rate
	== when condition1 has lower latency gradient, lower latency ratio,  lower packet loss, sending rate of condition 1 should be eventually (after 10 steps) higher than output of condition 2

'''


def create_next_step(input):
    next_input = input.copy()
    next_input = np.roll(next_input, -1, axis=1)
    next_input[0][9] = random.random() * 0.001
    next_input[1][9] = 1 + random.random() * 0.001
    next_input[2][9] = 1
    return next_input


def get_inputs_array(spec_type):
    if spec_type == 101:
        gene_inputs = np.zeros((3, HISTORY))
        for i in range(HISTORY):
            gene_inputs[0][i] = random.random() * 0.005
            gene_inputs[1][i] = 1 + random.random() * 0.005
            gene_inputs[2][i] = 1
    if spec_type == 102:
        gene_inputs = np.zeros((3, HISTORY))
        for i in range(HISTORY):
            gene_inputs[0][i] = 0.5 + random.random() * 0.5
            gene_inputs[1][i] = 5 + random.random() * 5
            gene_inputs[2][i] = 2 + random.random()
    if spec_type == 2:
        gene_inputs = np.zeros((3, HISTORY))
        for i in range(HISTORY):
            gene_inputs[0][i] = random.random() * 0.01
            gene_inputs[1][i] = 1 + random.random() * 0.01
            gene_inputs[2][i] = 2 + random.random()
    if spec_type == 3:
        myinput = get_inputs_array(101)
        myinput2 = myinput.copy()
        myinput2[0][9] = myinput2[0][9] + 0.1
        myinput2[1][9] = myinput2[1][9] + 0.1
        gene_inputs = np.concatenate([myinput, myinput2])
    if spec_type == 4:
        myinput = get_inputs_array(102)
        myinput2 = create_next_step(myinput)
        myinput3 = create_next_step(myinput2)
        myinput4 = create_next_step(myinput3)
        myinput5 = create_next_step(myinput4)
        gene_inputs = np.concatenate([myinput, myinput2, myinput3, myinput4, myinput5])
    return (gene_inputs)


def gene_spec():
    spec_number = len(SPEC_TYPES)
    for i in range(spec_number):
        input_arr = np.empty((SPEC_ARRAY_NUM, SPEC_ARRAY_LENGTH[i]))
        for j in range(SPEC_ARRAY_NUM):
            X = get_inputs_array(SPEC_TYPES[i]).flatten()
            input_arr[j] = X
        np.save(DIR + f'/aurora_fixedInput_{SPEC_TYPES[i]}.npy', input_arr)


def gen_index():
    for i in range(0, len(SPEC_TYPES)):
        index_arr = np.empty(3000)
        for j in range(3000):
            # first number is model number, second is range number
            index_arr[j] = 200000 + j

        np.save(DIR + f'/aurora_index_{SPEC_TYPES[i]}.npy', index_arr)


def main():
    random.seed(RANDOMSEED)
    if not os.path.exists(DIR):
        os.makedirs(DIR)
    gene_spec()
    gen_index()


if __name__ == "__main__":
    main()
