import argparse
import numpy as np
import os
import torch
from model.model import Model
from utils.wrappers import NormalizeWrapper, ResizeWrapper, ImgWrapper
from gym_duckietown.envs import DuckietownEnv
from pdb import set_trace

REF_VELOCITY = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser()
    # Do not change this
    parser.add_argument('--max_steps', type=int, default=2000, help='max_steps')

    # You should set them to different map name and seed accordingly
    parser.add_argument('--map-name', default='map1')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    args = parser.parse_args()

    model = Model()

    try:
        state_dict = torch.load(os.path.join('model', 'model.pt'), map_location=device)
        model.load_state_dict(state_dict)
    except:
        print('failed to load model')
        exit()
    
    model.eval().to(device)

    env = DuckietownEnv(
        map_name = args.map_name,
        domain_rand = False,
        draw_bbox = False,
        max_steps = args.max_steps,
        seed = args.seed
    )

    env = ResizeWrapper(env)
    env = NormalizeWrapper(env)
    env = ImgWrapper(env)
    obs = env.reset()

    actions = []
    total_reward = 0

    while True:
        obs = torch.FloatTensor(obs).to(device).unsqueeze(0)
        steering = model(obs)
        steering = steering.squeeze().data.cpu().numpy().item()

        action = (REF_VELOCITY, steering)
        actions.append(action)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        env.render()

        if done:
            env.close()
            break

    print("Total Reward", total_reward)

    # dump the controls using numpy
    result_path = os.path.join('result', '{}_seed{}.txt'.format(args.map_name, args.seed))
    if not os.path.exists(os.path.dirname(result_path)):
        try:
            os.makedirs(os.path.dirname(result_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    np.savetxt(result_path, actions, delimiter=',')

if __name__ == '__main__':
    main()