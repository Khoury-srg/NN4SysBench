import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
from spark_env.env import Environment
from average_reward import *
from compute_baselines import *
from compute_gradients import *
from actor_agent import ActorAgent

RANDOM_SEED = 20
MODEL_DIR = "./models"
RESULT_DIR = "./result"
HIDDEN_LAYERS = 2
NUM_AGENTS = 1
model_save_interval = 100
start_ep = 0
model_file = "./models/model_exec50_ep_" + str(start_ep)
learning_rate=0.001


def invoke_model(actor_agent, obs, exp):
    # parse observation
    job_dags, source_job, num_source_exec, \
    frontier_nodes, executor_limits, \
    exec_commit, moving_executors, action_map = obs

    if len(frontier_nodes) == 0:
        # no action to take
        return None, num_source_exec

    # invoking the learning model
    node_act, job_act, \
    node_act_probs, job_act_probs, \
    node_inputs, job_inputs, \
    node_valid_mask, job_valid_mask, \
    gcn_mats, gcn_masks, summ_mats, \
    running_dags_mat, dag_summ_backward_map, \
    exec_map, job_dags_changed = \
        actor_agent.invoke_model(obs)


    if sum(node_valid_mask[0, :]) == 0:
        # no node is valid to assign
        return None, num_source_exec

    # node_act should be valid
    try:
        assert node_valid_mask[0, node_act[0]] == 1
    except:
        return -1, -1

    # parse node action
    node = action_map[node_act[0].item()]

    # find job index based on node
    job_idx = job_dags.index(node.job_dag)

    # job_act should be valid
    assert job_valid_mask[0, job_act[0, job_idx] + \
                          len(actor_agent.executor_levels) * job_idx] == 1

    # find out the executor limit decision
    if node.job_dag is source_job:
        agent_exec_act = actor_agent.executor_levels[
                             job_act[0, job_idx]] - \
                         exec_map[node.job_dag] + \
                         num_source_exec
    else:
        agent_exec_act = actor_agent.executor_levels[
                             job_act[0, job_idx]] - exec_map[node.job_dag]

    # parse job limit action
    use_exec = min(
        node.num_tasks - node.next_task_idx - \
        exec_commit.node_commit[node] - \
        moving_executors.count(node),
        agent_exec_act, num_source_exec)

    # for storing the action vector in experience
    node_act_vec = torch.zeros(node_act_probs.shape)
    node_act_vec[0, node_act[0]] = 1

    # for storing job index
    job_act_vec = torch.zeros(job_act_probs.shape)
    job_act_vec[0, job_idx, job_act[0, job_idx]] = 1

    # store experience
    exp['node_inputs'].append(node_inputs)
    exp['job_inputs'].append(job_inputs)
    exp['summ_mats'].append(summ_mats)
    exp['running_dag_mat'].append(running_dags_mat)
    exp['node_act_vec'].append(node_act_vec)
    exp['job_act_vec'].append(job_act_vec)
    exp['node_valid_mask'].append(node_valid_mask)
    exp['job_valid_mask'].append(job_valid_mask)
    exp['job_state_change'].append(job_dags_changed)

    if job_dags_changed:
        exp['gcn_mats'].append(gcn_mats)
        exp['gcn_masks'].append(gcn_masks)
        exp['dag_summ_back_mat'].append(dag_summ_backward_map)

    return node, use_exec


def main(random_seed=RANDOM_SEED):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # create result and model folder
    create_folder_if_not_exists(RESULT_DIR)
    create_folder_if_not_exists(MODEL_DIR)

    # set up actor agent
    actor_agent = ActorAgent(
        args.node_input_dim, args.job_input_dim,
        args.hid_dims, args.output_dim, args.max_depth,
        range(1, args.exec_cap + 1), lr=learning_rate)

    if start_ep > 0:
        actor_agent.restore_models(model_file)

    # store average reward for computing differential rewards
    avg_reward_calculator = AveragePerStepReward(
        args.average_reward_storage_size)

    # initialize entropy parameters
    entropy_weight = args.entropy_weight_init

    # initialize episode reset probability
    reset_prob = args.reset_prob

    # ---- start training process ----
    for ep in range(start_ep + 1, args.num_ep):
        cur_time = time.time()
        last_time = cur_time

        # generate max time stochastically based on reset prob
        max_time = generate_coin_flips(reset_prob)

        # storage for advantage computation
        all_rewards, all_diff_times, all_times, \
        all_num_finished_jobs, all_avg_job_duration, \
        all_reset_hit, = [], [], [], [], [], []

        # set up environment
        env = Environment()

        # reset environment
        env.seed(random_seed)
        env.reset(max_time=max_time)

        # set up storage for experience
        exp = {'node_inputs': [], 'job_inputs': [],
               'gcn_mats': [], 'gcn_masks': [],
               'summ_mats': [], 'running_dag_mat': [],
               'dag_summ_back_mat': [],
               'node_act_vec': [], 'job_act_vec': [],
               'node_valid_mask': [], 'job_valid_mask': [],
               'reward': [], 'wall_time': [],
               'job_state_change': []}

        obs = env.observe()
        done = False

        exp['wall_time'].append(env.wall_time.curr_time)

        while not done:
            node, use_exec = invoke_model(actor_agent, obs, exp)
            if node==-1:
                break

            obs, reward, done = env.step(node, use_exec)

            if node is not None:
                # valid action, store reward and time
                exp['reward'].append(reward)
                exp['wall_time'].append(env.wall_time.curr_time)
            elif len(exp['reward']) > 0:
                # Note: if we skip the reward when node is None
                # (i.e., no available actions), the sneaky
                # agent will learn to exhaustively pick all
                # nodes in one scheduling round, in order to
                # avoid the negative reward
                exp['reward'][-1] += reward
                exp['wall_time'][-1] = env.wall_time.curr_time
        if node==-1:
            continue

        # report reward signals to master
        assert len(exp['node_inputs']) == len(exp['reward'])

        for i in range(NUM_AGENTS):
            result = [exp['reward'], exp['wall_time'],
                      len(env.finished_job_dags),
                      np.mean([j.completion_time - j.start_time
                               for j in env.finished_job_dags]),
                      env.wall_time.curr_time >= env.max_time]

            batch_reward, batch_time, \
            num_finished_jobs, avg_job_duration, \
            reset_hit = result

            diff_time = np.array(batch_time[1:]) - \
                        np.array(batch_time[:-1])

            all_rewards.append(batch_reward)
            all_diff_times.append(diff_time)
            all_times.append(batch_time[1:])
            all_num_finished_jobs.append(num_finished_jobs)
            all_avg_job_duration.append(avg_job_duration)
            all_reset_hit.append(reset_hit)

            avg_reward_calculator.add_list_filter_zero(
                batch_reward, diff_time)

        # compute differential reward
        all_cum_reward = []
        avg_per_step_reward = avg_reward_calculator.get_avg_per_step_reward()
        for i in range(NUM_AGENTS):
            if args.diff_reward_enabled:
                # differential reward mode on
                rewards = np.array([r - avg_per_step_reward * t for \
                                    (r, t) in zip(all_rewards[i], all_diff_times[i])])
            else:
                # regular reward
                rewards = np.array([r for \
                                    (r, t) in zip(all_rewards[i], all_diff_times[i])])

            cum_reward = discount(rewards, args.gamma)

            all_cum_reward.append(cum_reward)

        # compute baseline
        baselines = get_piecewise_linear_fit_baseline(all_cum_reward, all_times)

        # give worker back the advantage
        for i in range(NUM_AGENTS):
            batch_adv = all_cum_reward[i] - baselines[i]
            number = len(batch_adv)
            batch_adv = torch.tensor(batch_adv)
            batch_adv = torch.reshape(batch_adv, [number, 1])

        # compute gradients
        actor_loss, loss = compute_actor_gradients(actor_agent, exp, batch_adv, entropy_weight)

        actor_agent.apply_gradients(actor_loss)

        print(f"actor loss: {actor_loss}  adv_loss:{loss[0]}   entropy_loss:{loss[1]}")

        # decrease entropy weight
        entropy_weight = decrease_var(entropy_weight,
                                      args.entropy_weight_min, args.entropy_weight_decay)

        # decrease reset probability
        reset_prob = decrease_var(reset_prob,
                                  args.reset_prob_min, args.reset_prob_decay)

        if ep % model_save_interval == 0:
            actor_agent.save_model("./models/model_exec50_ep_" + str(ep))


if __name__ == '__main__':
    main()
