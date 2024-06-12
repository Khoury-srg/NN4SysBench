"""
This code is for illustration purpose only.
Use multi_agent.py for better performance and speed.
"""

import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.distributions import Categorical

import env
import model as a3c
import load_trace
import torch.optim as optim

import warnings

warnings.filterwarnings('ignore')
import logging

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]  # Kbps
HD_REWARD = [1, 2, 3, 12, 15, 20]
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000000
GRADIENT_BATCH_SIZE = 16
SUMMARY_DIR = './results'
LOG_FILE = './results/log'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
NN_MODEL = None
max_epoch = 1000
REWARD_LIST = ['linear', 'log', 'hd']
REWARD = REWARD_LIST[1]
MODEL_LIST = ['small', 'mid', 'big']
MODEL = MODEL_LIST[1]


def main():
    epoc_set = []
    qoe_set = []

    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    all_cooked_time, all_cooked_bw, _ = load_trace.load_trace()

    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)

    with open(LOG_FILE, 'w') as log_file:

        if MODEL == 'mid':
            actor = a3c.ActorNetwork_mid(state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                         learning_rate=ACTOR_LR_RATE)
        if MODEL == 'big':
            actor = a3c.ActorNetwork_big(state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                         learning_rate=ACTOR_LR_RATE)
        if MODEL == 'small':
            actor = a3c.ActorNetwork_small(state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                           learning_rate=ACTOR_LR_RATE)

        critic = a3c.CriticNetwork(state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)


        # restore neural net parameters
        nn_model = NN_MODEL
        if nn_model is not None:  # nn_model is the path to file
            actor.load_state_dict(torch.load(nn_model))
            print("Model restored.")

        epoch = 0
        time_stamp = 0

        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY

        action_vec = np.zeros(A_DIM)
        action_vec[bit_rate] = 1

        s_batch = [np.zeros((S_INFO, S_LEN))]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []

        actor_loss_batch = []
        critic_loss_batch = []

        while True:  # serve video forever
            # the action is from the last decision
            # this is to make the framework similar to the real
            delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = \
                net_env.get_video_chunk(bit_rate)

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            # -- linear reward --
            # reward is video quality - rebuffer penalty - smoothness
            if REWARD == REWARD_LIST[0]:
                REBUF_PENALTY = 4.3
                reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                         - REBUF_PENALTY * rebuf \
                         - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                                   VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

            # -- log scale reward --
            if REWARD == REWARD_LIST[1]:
                REBUF_PENALTY = 2.66
                log_bit_rate = np.log(VIDEO_BIT_RATE[bit_rate] / float(VIDEO_BIT_RATE[-1]))
                log_last_bit_rate = np.log(VIDEO_BIT_RATE[last_bit_rate] / float(VIDEO_BIT_RATE[-1]))

                reward = log_bit_rate \
                         - REBUF_PENALTY * rebuf \
                         - SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate)

            # -- HD reward --
            if REWARD == REWARD_LIST[2]:
                REBUF_PENALTY = 8
                reward = HD_REWARD[bit_rate] \
                         - REBUF_PENALTY * rebuf \
                         - SMOOTH_PENALTY * np.abs(HD_REWARD[bit_rate] - HD_REWARD[last_bit_rate])
            r_batch.append(reward)

            last_bit_rate = bit_rate

            # retrieve previous state
            if len(s_batch) == 0:
                state = [np.zeros((S_INFO, S_LEN))]
            else:
                state = np.array(s_batch[-1], copy=True)

            # dequeue history record
            state = np.roll(state, -1, axis=1)

            # this should be S_INFO number of terms
            state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
            state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  # 10 sec
            state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
            state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec 
            state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  # mega byte
            state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
            # get the next action
            with torch.no_grad():
                action_prob = actor.forward(torch.from_numpy(np.reshape(state, (S_INFO, S_LEN))))
            m = Categorical(action_prob)
            action = m.sample().item()
            bit_rate = action
            # Note: we need to discretize the probability into 1/RAND_RANGE steps,
            # because there is an intrinsic discrepancy in passing single state and batch states

            entropy_record.append(a3c.compute_entropy(action_prob[0]))

            # log time_stamp, bit_rate, buffer_size, reward
            log_file.write(str(time_stamp) + '\t' +
                           str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                           str(buffer_size) + '\t' +
                           str(rebuf) + '\t' +
                           str(video_chunk_size) + '\t' +
                           str(delay) + '\t' +
                           str(reward) + '\n')
            log_file.flush()

            if len(r_batch) >= TRAIN_SEQ_LEN or end_of_video:  # do training once
                actor_gradient, critic_gradient, td_batch = \
                    a3c.compute_loss(s_batch=np.stack(s_batch[1:], axis=0),  # ignore the first chuck
                                     a_batch=np.vstack(a_batch[1:]),  # since we don't have the
                                     r_batch=np.vstack(r_batch[1:]),  # control over it
                                     terminal=end_of_video, actor=actor, critic=critic)

                td_batch = td_batch.detach().numpy()
                td_loss = np.sum(td_batch)

                actor_loss_batch.append(actor_gradient)
                critic_loss_batch.append(critic_gradient)

                if len(actor_loss_batch) >= GRADIENT_BATCH_SIZE:

                    assert len(actor_loss_batch) == len(critic_loss_batch)

                    optimizer_actor = optim.RMSprop(actor.parameters(), lr=ACTOR_LR_RATE)
                    optimizer_critic = optim.RMSprop(critic.parameters(), lr=CRITIC_LR_RATE)

                    optimizer_critic.step()
                    optimizer_actor.step()
                    optimizer_actor.zero_grad()
                    optimizer_critic.zero_grad()

                    actor_loss_batch = []
                    critic_loss_batch = []


                    print("Epoch", epoch, "TD_loss", td_loss, "Avg_reward", np.sum(r_batch), "Avg_entropy",
                          np.sum(entropy_record))

                    epoc_set.append(epoch)
                    qoe_set.append(np.mean(r_batch))
                    entropy_record = []
                    epoch += 1

                    if epoch % MODEL_SAVE_INTERVAL == 0:
                        # Save the neural net parameters to disk.
                        save_path = SUMMARY_DIR + "/nn_model_ep_" + str(epoch) + ".pth"
                        torch.save(actor.state_dict(), save_path)
                        print("Model saved in file: %s" % save_path)

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]

            if end_of_video:
                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY  # use the default action here
                s_batch.append(np.zeros((S_INFO, S_LEN)))
                a_batch.append(action)

            else:
                s_batch.append(state)
                a_batch.append(action)

            if epoch > max_epoch:
                plt.plot(epoc_set, qoe_set, color="orange")
                plt.title('model qoe')
                plt.ylabel('qpe')
                plt.xlabel('epoch')
                plt.show()


if __name__ == '__main__':
    main()
