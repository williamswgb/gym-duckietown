import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
import argparse
from gym_duckietown.envs import DuckietownEnv

from pdb import set_trace

REF_VELOCITY = 0.2 # make the velocity constant

def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]
    polygons = np.array([[
        (0, height),
        (0, height/ 2),
        (width, height/ 2),
        (width, height),
    ]], dtype=np.int32)

    # Define a blank matrix that matches the image height/width.
    mask = np.zeros_like(image)

    # Fill inside the polygon
    cv2.fillPoly(mask, polygons, 255)

    # Returning the image only where mask pixels match
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def display_lines(image, lines, color=(255, 0, 0), thickness=10):
    line_image = np.zeros_like(image)
    if lines is not None and len(lines) > 0:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)

    line_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)

    return line_image

def canny(image):
    # Convert to grayscale here.
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Call Canny Edge Detection here.
    canny = cv2.Canny(blur, 150, 300)
    return canny

def make_points(image, line):
    height, width, _ = image.shape
    slope, intercept = line
    y1 = height  # bottom of the image
    y2 = int(y1 / 2)  # make points from middle of the image down

    # bound the coordinates within the image
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, line_segments):
    lane_lines = []
    """
    This function combines line segments into one or two lane lines
    If all line slopes are < 0: then we only have detected left lane
    If all line slopes are > 0: then we only have detected right lane
    """
    if line_segments is None:
        print('No line_segment segments detected')
        return lane_lines

    height, width, _ = image.shape
    left_fit = []
    right_fit = []

    # TODO: Remove boundary to handle horizontal line
    boundary = 1/3
    left_region_boundary = width * (1 - boundary)  # left lane line segment should be on left 1/3 of the screen
    right_region_boundary = width * boundary # right lane line segment should be on right 1/3 of the screen

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                print('skipping vertical line segment (slope=inf): %s' % line_segment)
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    if len(left_fit) > 0:
        left_fit_average = np.average(left_fit, axis=0)
        lane_lines.append(make_points(image, left_fit_average))

    if len(right_fit) > 0:
        right_fit_average = np.average(right_fit, axis=0)
        lane_lines.append(make_points(image, right_fit_average))

    # print('lane lines: %s' % lane_lines)

    return np.array(lane_lines)

def length_of_line_segment(line):
    x1, y1, x2, y2 = line
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def detect_line_segments(image):
    rho = 1
    theta = np.pi / 180
    threshold = 10
    min_line_length = 100
    max_line_gap = 10

    line_segments = cv2.HoughLinesP(
        image,
        rho=rho,
        theta=theta,
        threshold=threshold,
        lines=np.array([]),
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    return line_segments

def detect_lane(image):
    lane_image = np.copy(image)
    canny_image = canny(lane_image)
    cropped_image = region_of_interest(canny_image)

    line_segments = detect_line_segments(cropped_image)
    lane_lines = average_slope_intercept(lane_image, line_segments)

    line_image = display_lines(lane_image, line_segments)
    lane_line_image = display_lines(lane_image, lane_lines)

    return lane_lines, line_image, lane_line_image

def compute_new_steering(image, lane_lines, curr_steering):
    """ Find the steering angle based on lane line coordinate
        We assume that camera is calibrated to point to dead center
    """
    if len(lane_lines) == 0:
        print('No lane lines detected, do nothing')
        if curr_steering == 1e-18 or curr_steering == -1e-18:
            return 0
        elif curr_steering > 0:
            return max(curr_steering * 0.7, 1e-18)
        elif curr_steering < 0:
            return min(curr_steering * 0.7, -1e-18)
        else:
            return 0

    height, width, _ = image.shape
    lane_lines = lane_lines.tolist()
    if len(lane_lines) == 1:
        # print('Only detected one lane line, just follow it. %s' % lane_lines[0])
        x1, y1, x2, y2 = lane_lines[0]
        x_offset = x2 - x1
        y_offset = y2 - y1
    else:
        _, _, left_x2, _ = lane_lines[0]
        _, _, right_x2, _ = lane_lines[1]
        mid = int(width / 2)
        x_offset = (left_x2 + right_x2) / 2 - mid
        # find the steering angle, which is angle between navigation direction to end of center line
        y_offset = int(height / 2)

    angle_to_mid_radian = math.atan(x_offset / y_offset)  # angle (in radian) to center vertical line
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)

    print('New steering: %s' % angle_to_mid_deg)
    return angle_to_mid_radian

def stabilize_steering(curr_steering, new_steering, num_of_lane_lines, max_angle_deviation_two_lines=5, max_angle_deviation_one_lane=1):
    """
    Using last steering angle to stabilize the steering angle
    This can be improved to use last N angles, etc
    if new angle is too different from current angle, only turn by max_angle_deviation degrees
    """
    if num_of_lane_lines == 2:
        # if both lane lines detected, then we can deviate more
        max_angle_deviation = max_angle_deviation_two_lines
    else:
        # if only one lane detected, don't deviate too much
        max_angle_deviation = max_angle_deviation_one_lane

    angle_deviation = new_steering - curr_steering
    if abs(angle_deviation) > max_angle_deviation:
        stabilized_steering = int(curr_steering + max_angle_deviation * angle_deviation / abs(angle_deviation))
    else:
        stabilized_steering = new_steering

    print('Proposed steering: %s, stabilized steering: %s' % (new_steering, stabilized_steering))

    return stabilized_steering

def save_image(obs, idx, folder_name='test'):
    data_path = os.path.join('samples', 'data', folder_name, '{}.png'.format(idx))
    if not os.path.exists(os.path.dirname(data_path)):
        try:
            os.makedirs(os.path.dirname(data_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    cv2.imwrite(data_path, obs)

if __name__ == '__main__':
    # declare the arguments
    parser = argparse.ArgumentParser()

    # Do not change this
    parser.add_argument('--max_steps', type=int, default=2000, help='max_steps')

    # You should set them to different map name and seed accordingly
    parser.add_argument('--map_name', default='map1')
    parser.add_argument('--seed', type=int, default=11, help='random seed')
    args = parser.parse_args()

    env = DuckietownEnv(
        map_name=args.map_name,
        domain_rand=False,
        draw_bbox=False,
        max_steps=args.max_steps,
        seed=args.seed
    )

    obs = env.reset()
    env.render()

    actions = []
    total_reward = 0
    steering = 0

    while True:
        lane_lines, line_image, lane_line_image = detect_lane(obs)
        new_steering = compute_new_steering(obs, lane_lines, steering)
        steering = stabilize_steering(steering, new_steering, len(lane_lines))

        action = (REF_VELOCITY, steering)
        obs, reward, done, info = env.step(action)
        actions.append(action)
        total_reward += reward
        env.render()

        print('Steps = %s, Timestep Reward=%.3f, Total Reward=%.3f' % (env.step_count, reward, total_reward))

        if done:
            env.close()
            break

    print("Total Reward", total_reward)

    # # dump the controls using numpy
    # result_path = os.path.join('result', 'test_{}_seed{}.txt'.format(args.map_name, args.seed))
    # if not os.path.exists(os.path.dirname(result_path)):
    #     try:
    #         os.makedirs(os.path.dirname(result_path))
    #     except OSError as exc: # Guard against race condition
    #         if exc.errno != errno.EEXIST:
    #             raise
    # np.savetxt(result_path, actions, delimiter=',')