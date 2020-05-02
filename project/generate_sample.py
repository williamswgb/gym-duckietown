import numpy as np
import cv2
import os
from utils.teacher import PurePursuitExpert
from utils.wrappers import ResizeWrapper
from gym_duckietown.envs import DuckietownEnv
from pdb import set_trace

def get_samples_from_map(map=1, seed=11, max_steps=2000):
    steerings = []
    observations = []

    env = DuckietownEnv(
        map_name='map{}'.format(map),
        domain_rand=False,
        draw_bbox=False,
        max_steps=max_steps,
        seed=seed
    )
    env = ResizeWrapper(env)

    obs = env.reset()
    observations.append(obs)
    expert = PurePursuitExpert(env)

    for i in range(max_steps):
        action = expert.predict(obs)
        obs, _, _, _ = env.step(action)
        observations.append(obs)
        steerings.append(action[1])

    env.close()

    return observations[0:max_steps], steerings

def generate_samples():
    data = []
    label = []

    for i in range(5):
        observations, steerings = get_samples_from_map(map=i+1)
        data.extend(observations)
        label.extend(steerings)

        for idx, o in enumerate(observations):
            data_path = os.path.join('samples', 'data', str(i+1), '{}.png'.format(idx))
            if not os.path.exists(os.path.dirname(data_path)):
                try:
                    os.makedirs(os.path.dirname(data_path))
                except OSError as exc: # Guard against race condition
                    if exc.errno != errno.EEXIST:
                        raise
            cv2.imwrite(data_path, o)

    label_path = os.path.join('samples', 'label.txt')
    if not os.path.exists(os.path.dirname(label_path)):
        try:
            os.makedirs(os.path.dirname(label_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    np.savetxt(label_path, label)

    print('Generated data: {}'.format(len(data)))
    print('Generated label: {}'.format(len(label)))

if __name__ == '__main__':
    generate_samples()