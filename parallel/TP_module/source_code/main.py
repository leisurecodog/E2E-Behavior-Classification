#!/usr/bin/env python3 

import numpy as np

from copy import deepcopy
import torch
# import gym

# from normalized_env import NormalizedEnv
from evaluator import Evaluator
from ddpg import DDPG
from util import *
import env as Env
import parser
# gym.undo_logger_setup()

def test(num_episodes, agent, env, evaluate, model_path, visualize=True, debug=False):

    agent.load_weights(model_path)
    agent.is_training = False
    agent.eval()
    policy = lambda x: agent.select_action(x, decay_epsilon=False)
    res = []
    res_fde = []
    for i in range(num_episodes):
        validate_reward, fde = evaluate(env, policy, debug=debug, visualize=visualize, save=False)
        res.append(validate_reward)
        res_fde.append(fde)
        if debug: prYellow('[Evaluate] #{}: mean_reward:{}'.format(i, validate_reward))
    print(np.sum(res) / len(res))
    print(np.sum(res_fde) / len(res_fde))

def train(num_iterations, agent, env,  evaluate, validate_steps, output, max_episode_length=None, debug=False):
    from tensorboardX import SummaryWriter 
    writer = SummaryWriter(comment=output)
    agent.is_training = True
    step = episode = episode_steps = 0
    episode_reward = 0.
    observation = None
    best_valid = -np.inf
    while step < num_iterations:
        # reset if it is the start of episode
        if observation is None:
            observation = deepcopy(env.reset())
            # print(observation)
            agent.reset(observation)

        # agent pick action ...
        if step <= args.warmup:
            action = agent.random_action()
        else:
            action = agent.select_action(observation)
        
        # env response with next_observation, reward, terminate_info
        observation2, reward, done, info = env.step(action, step)
        # print(observation2)
        observation2 = deepcopy(observation2)
        if max_episode_length and episode_steps >= max_episode_length -1:
            done = True

        # agent observe and update policy
        agent.observe(reward, observation2, done)
        if done:
            agent.update_policy()
        
        # [optional] evaluate
        # if evaluate is not None and validate_steps > 0 and step % validate_steps == 0:
        if done and episode % 100 == 0:
            # env_eval = Env.environment()
            policy = lambda x: agent.select_action(x, decay_epsilon=False)
            validate_reward, _ = evaluate(env, policy, debug=False, visualize=False)
            if validate_reward > best_valid:
                best_valid = validate_reward
                agent.save_model(output, best=True)

            writer.add_scalar('Valid Reward', validate_reward, episode)

            if debug: 
                prYellow('[Evaluate] Step_{:07d}: mean_reward:{}'.format(step, validate_reward))
        # [optional] save intermideate model
        if step % int(num_iterations/3) == 0:
            agent.save_model(output)

        # update 
        step += 1
        episode_steps += 1
        episode_reward += reward
        observation = deepcopy(observation2)
        
        if done: # end of episode
            episode_reward /= episode_steps
            if debug: prGreen('#{}: episode_reward:{} steps:{}, tf-ratio:{},{}'.format(episode, episode_reward, step, env.teacher_forcing_ratio, env.prob))
            writer.add_scalar('Training Reward', episode_reward, episode)
            agent.memory.append(
                observation,
                agent.select_action(observation),
                0., False
            )
            # reset
            observation = None
            episode_steps = 0
            episode_reward = 0.
            episode += 1
    env.mode = 'test'
    test(args.validate_episodes, agent, env, evaluate, args.resume,
            visualize=True, debug=args.debug)

if __name__ == "__main__":

    args = parser.get_parser()
    args.output = get_output_folder(args.output, args.env)
    if args.resume == 'default':
        # args.resume = 'output/{}-run0'.format(args.env)
        args.resume = args.output

    # env = NormalizedEnv(gym.make(args.env))
    env = Env.environment(data_select=args.mode, args=args)
    # if args.seed > 0:
    #     np.random.seed(args.seed)
    #     env.seed(args.seed)
    nb_states = env.nb_s
    nb_actions = env.nb_a

    agent = DDPG(nb_states, nb_actions, args)
    # print(args.actor)
    evaluate = Evaluator(args.validate_episodes,
        args.validate_steps, args.output, max_episode_length=args.max_episode_length)
    # evaluate = None

    if args.mode == 'train':
        # from demo import demo
        
        train(args.train_iter, agent, env, evaluate, 
            args.validate_steps, args.output, max_episode_length=args.max_episode_length, debug=args.debug)

    elif args.mode == 'test':
        test(args.validate_episodes, agent, env, evaluate, args.resume,
            visualize=True, debug=args.debug)
    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))
    
