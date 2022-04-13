import gym
import colossalai
import torch
import time
import numpy as np
from torch.nn import MSELoss
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger, disable_existing_loggers
from colossalai.utils import print_rank_0, save_checkpoint, load_checkpoint
from tensorboardX import SummaryWriter

from model.agent import naive_DQN
from model.networks import Cart_Pole_Network
from model.utils import get_epsilon_by_step, ReplayBuffer
from model.utils import init_atari_env, Atari_envs, Gym_envs

def main():
    parser = colossalai.get_default_parser()
    parser.add_argument('--from_torch', default=False, action='store_true')
    args = parser.parse_args()
    disable_existing_loggers()
    if args.from_torch:
        colossalai.launch_from_torch(config=args.config)
    else:
        colossalai.launch_from_slurm(config=args.config,
                                     host=args.host,
                                     port=29500,
                                     seed=42)

    logger = get_dist_logger()

    # initial environment
    env_id = gpc.config.environment
    logger.info(f'Environment {env_id} initialization begins', ranks=[0])
    if env_id in Gym_envs:
        env = gym.make(env_id).unwrapped
    elif env_id in Atari_envs:
        env = init_atari_env(env_id)
    else:
        logger.info(f'Environment {env_id} is not supported', ranks=[0])
        raise NotImplementedError

    # initial tensorboard writer for recording loss and reward
    log_path = gpc.config.get('log_path', f'./logs/{env_id}')
    writer = SummaryWriter(log_path)

    num_actions = env.action_space.n

    logger.info('Building model', ranks=[0])
    network = gpc.config.model.get('type', Cart_Pole_Network)(env.observation_space.shape, num_actions)
    optimizer = gpc.config.optimizer.pop('type')(network.parameters(), **gpc.config.optimizer)

    # get loss function for colossalai engine
    criterion = getattr(gpc.config, 'loss', None)
    if criterion is not None:
        criterion = criterion.type()
    else:
        criterion = MSELoss()

    # use lr_scheduler to decay learning rate by episode
    lr_scheduler = None
    if hasattr(gpc.config, 'lr_scheduler'):
        lr_scheduler = gpc.config.lr_scheduler.pop('type')(**gpc.config.lr_scheduler)

    # load checkpoint from last training here
    # load_checkpoint(checkpoint_path, network, optimizer, lr_scheduler)

    # initialize colossalai engine for both networks
    eval_engine, _, _, _ = colossalai.initialize(network, optimizer, criterion)
    target_engine, _, _, _ = colossalai.initialize(network, optimizer, criterion)
    # target net should have same mode as eval net for updating params, but no backward.
    eval_engine.train()
    target_engine.train()

    logger.info('Build DQN agent', ranks=[0])
    dqn = naive_DQN(eval_engine, target_engine, gpc.config.batch_size, num_actions, gpc.config.memory_capacity, gpc.config.gamma)

    logger.info('Collecting Experience....', ranks=[0])
    global_step = 0
    episode = 0
    ep_reward_list = []
    start_time = time.time()
    while global_step <= gpc.config.total_step:
        ep_reward = 0
        learn_step_counter = 0
        step = 0
        loss = None
        # episode counter start from 1
        episode += 1

        state = env.reset()
        for step in range(gpc.config.max_steps):
            # get epsilon by step for epsilon greedy
            epsilon = get_epsilon_by_step(global_step, gpc.config.epsilon_start,
                                          gpc.config.epsilon_final, gpc.config.epsilon_decay)
            # agent using eval_net choose actions for this state
            action = dqn.choose_action(state, epsilon)
            # update transition and store in replay buffer
            next_state, reward, done, info = env.step(action)
            transition = (state, action, reward, next_state, done)
            dqn.store_transition(transition)
            ep_reward += reward

            # if enough memory for buffer has been stored, learning would begin
            if len(dqn.replay_buffer) > gpc.config.pre_step:
                # update parameters for target network, only at first rank
                if learn_step_counter % gpc.config.update_iteration == 0 and gpc.get_global_rank() == 0:
                    dqn.update()
                learn_step_counter += 1
                # single learning step for eval_net
                loss = dqn.step_learn()
                global_step += 1
            elif step % 100 == 0:
                logger.info(f'Replay buffer memory {len(dqn.replay_buffer)} is too less', ranks=[0])

            # move to next episode if simulation is over
            if done or global_step > gpc.config.total_step:
                break
            state = next_state

        # decay learning rate
        if lr_scheduler is not None:
            lr_scheduler.step()

        # record log and info
        if gpc.get_global_rank() == 0:
            ep_reward_list.append(ep_reward)
            if loss is not None:
                writer.add_scalar('loss/value_loss', loss, global_step)
                writer.add_scalar('reward/episode_reward', ep_reward, episode)
            if episode % 10 == 0 and loss is not None:
                print_rank_0(f"episode: {episode}, step: {step}, max reward in 10 episode: {np.max(ep_reward_list[-10:])}, "
                             f"average reward of 10 episode: {round(np.mean(ep_reward_list[-10:]), 3)},"
                             f"loss {loss}, total_step: {global_step}")

    end_time = time.time()
    logger.info(f"training cost time: {end_time-start_time}", ranks=[0])
    # save checkpoint for this training here (please create your empty ckpt-file before)
    # if gpc.get_global_rank() == 0:
    #     save_checkpoint(file=gpc.config.get('save_checkpoint_path', f'tmp/{env_id}_ckpt'), epoch=episode, model=network)

    # avoid cuda error of torch dp
    gpc.destroy()


if __name__ == '__main__':
    main()
