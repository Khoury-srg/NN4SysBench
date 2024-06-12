import random

import numpy as np
import math

import torch

import env_benchmark as env

S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
BUFFER_NORM_FACTOR = 10.0
M_IN_K = 1000.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
RANDOM_SEED = 2024

SMOOTH_PENALTY = 1
BITRATE_LEVELS = 6


def create_bw(condition="good", length=1000, random_seed=RANDOM_SEED):
    random.seed(random_seed)
    time = []
    bandwidth = []
    cur_time = 0

    for i in range(length):
        if condition == "good":
            cur_bw = 10 + 0.1 * random.random()
        if condition == "bad":
            cur_bw = 1 * random.random()
        time.append(cur_time)
        cur_time += random.random() * 0.02 + 1
        bandwidth.append(cur_bw)

    return time, bandwidth


def create_video_size(largest_size=2400000, length=100, random_seed=RANDOM_SEED):
    random.seed(random_seed)
    video_size = {}
    largest_size = largest_size
    for bitrate in range(BITRATE_LEVELS):
        video_size[bitrate] = []
        for i in range(length):
            video_size[bitrate].append(largest_size / math.pow(2, 5 - bitrate) + 100000 * random.random())
    return video_size


def get_inputs(all_cooked_time, all_cooked_bw, video_size, max_chunk_size=2400000,
               chunk_num=6, buffer=10, difficulty=0, condition="good", step="multiple"):
    if condition == "good":
        action = 5
    if condition == "bad":
        action = 2

    net_env = env.Environment(cooked_time=all_cooked_time,
                              cooked_bw=all_cooked_bw,
                              video_size=video_size)

    state = np.zeros((S_INFO, S_LEN))

    step = 40
    if condition == 'initial':  # initial state only run one time
        step = 0

    while True:
        bit_rate = action
        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, \
        end_of_video, video_chunk_remain = \
            net_env.get_video_chunk(bit_rate)

        state = np.roll(state, -1, axis=1)

        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
        state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
        state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
        state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

        step -= 1
        if step < 0:
            break
    return state


def get_inputs_array(spec_type, random_seed=RANDOM_SEED):
    random.seed(random_seed)
    if spec_type == 1:
        all_cooked_time, all_cooked_bw = create_bw(random_seed=random_seed)
        video_size = create_video_size(random_seed=random_seed)
        return get_inputs(all_cooked_time, all_cooked_bw, video_size, condition="good")
    if spec_type == 2:
        all_cooked_time, all_cooked_bw = create_bw(condition="bad", random_seed=random_seed)
        video_size = create_video_size(random_seed=random_seed)
        return get_inputs(all_cooked_time, all_cooked_bw, video_size, condition="bad")
    if spec_type == 3:
        all_cooked_time, all_cooked_bw = create_bw(condition="good", random_seed=random_seed)
        video_size = create_video_size(random_seed=random_seed)
        good_condition_input = get_inputs(all_cooked_time, all_cooked_bw, video_size, condition="good")

        all_cooked_time, all_cooked_bw = create_bw(condition="bad", random_seed=random_seed)
        video_size = create_video_size(random_seed=random_seed)
        bad_condition_input = get_inputs(all_cooked_time, all_cooked_bw, video_size, condition="bad")
        return np.concatenate([good_condition_input, bad_condition_input], 0)
