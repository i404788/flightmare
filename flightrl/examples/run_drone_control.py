#!/usr/bin/env python3
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()


import yaml
from pathlib import Path

import json
import os
import math
import argparse
import numpy as np
# import tensorflow as tf

#
# from stable_baselines import logger

#
# from rpg_baselines.common.policies import MlpPolicy
# from rpg_baselines.ppo.ppo2 import PPO2
# from rpg_baselines.ppo.ppo2_test import test_model
from rpg_baselines.envs import vec_env_wrapper as wrapper
# import rpg_baselines.common.util as U
#
from flightgym import QuadrotorEnv_v1


def configure_random_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
    np.random.seed(seed)
    # tf.set_random_seed(seed)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, default=1,
                        help="To train new model or simply test pre-trained model")
    parser.add_argument('--render', type=int, default=0,
                        help="Enable Unity Render")
    parser.add_argument('--save_dir', type=str, default=os.path.dirname(os.path.realpath(__file__)),
                        help="Directory where to save the checkpoints and training metrics")
    parser.add_argument('--seed', type=int, default=0,
                        help="Random seed")
    parser.add_argument('-w', '--weight', type=str, default='./saved/quadrotor_env.zip',
                        help='trained weight path')
    return parser


default_drone_cfg = yaml.safe_load("""
quadrotor_env:
   camera: no
   sim_dt: 0.02 
   max_t: 5.0
   add_camera: yes

quadrotor_dynamics:
  mass: 0.73
  arm_l: 0.17
  motor_omega_min: 150.0 # motor rpm min
  motor_omega_max: 3000.0 # motor rpm max
  motor_tau: 0.0001 # motor step response
  thrust_map: [1.3298253500372892e-06, 0.0038360810526746033, -1.7689986848125325]
  kappa: 0.016 # rotor drag coeff
  omega_max: [6.0, 6.0, 6.0]  # body rate constraint (x, y, z) 

rl:
  pos_coeff: -0.002        # reward coefficient for position 
  ori_coeff: -0.002        # reward coefficient for orientation
  lin_vel_coeff: -0.0002   # reward coefficient for linear velocity
  ang_vel_coeff: -0.0002   # reward coefficient for angular velocity
  act_coeff: -0.0002  # reward coefficient for control actions
""")


def main():
    args = parser().parse_args()
    cfg = yaml.safe_load(open(os.environ["FLIGHTMARE_PATH"] +
                           "/flightlib/configs/vec_env.yaml", 'r'))
    if not args.train:
        cfg["env"]["num_envs"] = 1
        cfg["env"]["num_threads"] = 1

    if args.render:
        cfg["env"]["render"] = True
    else:
        cfg["env"]["render"] = False

    print(cfg)
    cfg = json.dumps(cfg)
    Path('./quadenv.json').write_text(cfg)
    print(default_drone_cfg)
    Path('./quadrotor_env.json').write_text(json.dumps(default_drone_cfg))
    env = QuadrotorEnv_v1('./quadenv.json')
    env = wrapper.FlightEnvVec(env)

    # set random seed
    print(dir(env))
    configure_random_seed(args.seed, env=env)
    obs = env.reset()
    print(obs)

    # #
    # if args.train:
    #     # save the configuration and other files
    #     rsg_root = os.path.dirname(os.path.abspath(__file__))
    #     log_dir = rsg_root + '/saved'
    #     saver = U.ConfigurationSaver(log_dir=log_dir)
    #     model = PPO2(
    #         tensorboard_log=saver.data_dir,
    #         policy=MlpPolicy,  # check activation function
    #         policy_kwargs=dict(
    #             net_arch=[dict(pi=[128, 128], vf=[128, 128])], act_fun=tf.nn.relu),
    #         env=env,
    #         lam=0.95,
    #         gamma=0.99,  # lower 0.9 ~ 0.99
    #         # n_steps=math.floor(cfg['env']['max_time'] / cfg['env']['ctl_dt']),
    #         n_steps=250,
    #         ent_coef=0.00,
    #         learning_rate=3e-4,
    #         vf_coef=0.5,
    #         max_grad_norm=0.5,
    #         nminibatches=1,
    #         noptepochs=10,
    #         cliprange=0.2,
    #         verbose=1,
    #     )

    #     # tensorboard
    #     # Make sure that your chrome browser is already on.
    #     # TensorboardLauncher(saver.data_dir + '/PPO2_1')

    #     # PPO run
    #     # Originally the total timestep is 5 x 10^8
    #     # 10 zeros for nupdates to be 4000
    #     # 1000000000 is 2000 iterations and so
    #     # 2000000000 is 4000 iterations.
    #     logger.configure(folder=saver.data_dir)
    #     model.learn(
    #         total_timesteps=int(25000000),
    #         log_dir=saver.data_dir, logger=logger)
    #     model.save(saver.data_dir)

    # # # Testing mode with a trained weight
    # else:
    #     model = PPO2.load(args.weight)
    #     test_model(env, model, render=args.render)


if __name__ == "__main__":
    main()
