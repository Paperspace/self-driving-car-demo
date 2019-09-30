""" Credits comma.ai """

# !/usr/bin/env python
import argparse
import sys
import numpy as np
import h5py
import json
import tensorflow as tf
from pdb import set_trace as bp
import os
import time
import scipy.misc
from tensorflow import py_func

# ***** get perspective transform for images *****
from skimage import transform as trans

rsrc = \
    [[43.45456230828867, 118.00743250075844],
     [104.5055617352614, 69.46865203761757],
     [114.86050156739812, 60.83953551083698],
     [129.74572757609468, 50.48459567870026],
     [132.98164627363735, 46.38576532847949],
     [301.0336906326895, 98.16046448916306],
     [238.25686790036065, 62.56535881619311],
     [227.2547443287154, 56.30924933427718],
     [209.13359962247614, 46.817221154818526],
     [203.9561297064078, 43.5813024572758]]
rdst = \
    [[10.822125594094452, 1.42189132706374],
     [21.177065426231174, 1.5297552836484982],
     [25.275895776451954, 1.42189132706374],
     [36.062291434927694, 1.6376192402332563],
     [40.376849698318004, 1.42189132706374],
     [11.900765159942026, -2.1376192402332563],
     [22.25570499207874, -2.1376192402332563],
     [26.785991168638553, -2.029755283648498],
     [37.033067044190524, -2.029755283648498],
     [41.67121717733509, -2.029755283648498]]

tform3_img = trans.ProjectiveTransform()
tform3_img.estimate(np.array(rdst), np.array(rsrc))


def perspective_tform(x, y):
    p1, p2 = tform3_img((x, y))[0]
    return p2, p1


# ***** functions to draw lines *****
def draw_pt(img, x, y, color, sz=1):
    row, col = perspective_tform(x, y)
    if row >= 0 and row < img.shape[0] and \
            col >= 0 and col < img.shape[1]:
        img[int(row - sz):int(row + sz), int(col - sz):int(col + sz)] = color


def draw_path(img, path_x, path_y, color):
    for x, y in zip(path_x, path_y):
        draw_pt(img, x, y, color)


# ***** functions to draw predicted path *****

def calc_curvature(v_ego, angle_steers, angle_offset=0):
    deg_to_rad = np.pi / 180.
    slip_fator = 0.0014  # slip factor obtained from real data
    steer_ratio = 15.3  # from http://www.edmunds.com/acura/ilx/2016/road-test-specs/
    wheel_base = 2.67  # from http://www.edmunds.com/acura/ilx/2016/sedan/features-specs/

    angle_steers_rad = (angle_steers - angle_offset) * deg_to_rad
    curvature = angle_steers_rad / (steer_ratio * wheel_base * (1. + slip_fator * v_ego ** 2))
    return curvature


def calc_lookahead_offset(v_ego, angle_steers, d_lookahead, angle_offset=0):
    # *** this function returns the lateral offset given the steering angle, speed and the lookahead distance
    curvature = calc_curvature(v_ego, angle_steers, angle_offset)

    # clip is to avoid arcsin NaNs due to too sharp turns
    y_actual = d_lookahead * np.tan(np.arcsin(np.clip(d_lookahead * curvature, -0.999, 0.999)) / 2.)
    return y_actual, curvature


def draw_path_on(img, speed_ms, angle_steers, color=(0, 0, 255)):
    path_x = np.arange(0., 50.1, 0.5)
    path_y, _ = calc_lookahead_offset(speed_ms, angle_steers, path_x)
    draw_path(img, path_x, path_y, color)


def render_steering_frame(img, ground_truth, speed_ms, prediction):
    """
    Added by malo@goodailab.com
    Transforms the numpy representation of a road image and the steering / predicted steering into a perspective view of the path.
    Input:
    - img: numpy array, one camera frame
    - groud_truth: float, ground truth of the steering angle
    - speed_ms: float, ground truth for the speed
    - prediction: predicted steering angle, float
    Output:
    - numpy array, visualiztion of the road and the projected path of size (width,height,3)
    """
    predicted_steer = prediction
    angle_steers = ground_truth
    img = img.swapaxes(0, 2).swapaxes(0, 1)
    draw_path_on(img, speed_ms, - angle_steers / 10.0)
    draw_path_on(img, speed_ms, - predicted_steer / 10.0, (0, 255, 0))
    return (img)


def render_steering_frame_batch(img_batch, ground_truth_batch, speed_ms_batch, prediction_batch):
    """
    Added by malo@goodailab.com
    Applies render_steering_frame to a batch of images.
    Input:
    - img: numpy array, batch of camera frames. Numpy array of (batch_size, width,height,3)
    - groud_truth: batch of ground truth of the steering angle. Numpy array of (batch_size)
    - speed_ms: float, ground truth for the speed. Numpy array of size (batch_size)
    - prediction: predicted steering angle. Numpy array of size (batch_size)
    Output:
    - numpy array, visualiztion of the road and the projected path of size (width,height,3)
    """
    rendered_list = []
    for i in range(img_batch.shape[0]):
        rendered = render_steering_frame(img_batch[i, :, :, :], ground_truth_batch[i], speed_ms_batch[i], prediction_batch[i])
        rendered_list.append(rendered)
    rendered_list = np.asarray(rendered_list)

    assert len(rendered_list.shape) == 4  # sanity check
    assert rendered_list.shape[0] == img_batch.shape[0]  # sanity check

    return rendered_list.astype(np.float32)


def render_steering_tf(img_batch, ground_truth_batch, speed_ms_batch, prediction_batch):
    """
    Added by malo@goodailab.com.
    Bridge render_steering_frame_batch to tf
    """
    return py_func(render_steering_frame_batch, [img_batch, ground_truth_batch, speed_ms_batch, prediction_batch], tf.float32)


if __name__ == "__main__":
    import data_reader as dr

    DATA_DIR = os.path.expanduser('~/Documents/comma/comma-final/camera/training')
    print("Running tests")
    gen_train = dr.gen(DATA_DIR)
    X, angle, speed = gen_train.next()

    out = render_steering_frame_batch(X, angle, speed, np.repeat(3., X.shape[0]))
