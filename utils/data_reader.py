import numpy as np
import h5py
import time
import os
import logging
import traceback
import tensorflow as tf
import json
import glob

# Credits for this code and the data: comma.ai
# https://github.com/commaai/research
# Original License in LICENSE_COMMA


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class DataReader:

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.gen_line = self.datagen(time_len=1, batch_size=1, ignore_goods=False)
        self.first = True

    def concatenate(self, camera_names, time_len):
        logs_names = [x.replace('camera', 'labels') for x in camera_names]

        angle = []  # steering angle of the car
        speed = []  # steering angle of the car
        hdf5_camera = []  # the camera hdf5 files need to continue open
        c5x = []
        filters = []
        lastidx = 0

        for cword, tword in zip(camera_names, logs_names):
            try:
                with h5py.File(tword, "r") as t5:
                    c5 = h5py.File(cword, "r")
                    hdf5_camera.append(c5)
                    x = c5["X"]
                    c5x.append((lastidx, lastidx + x.shape[0], x))

                    speed_value = t5["speed"][:]
                    steering_angle = t5["steering_angle"][:]
                    idxs = np.linspace(0, steering_angle.shape[0] - 1, x.shape[0]).astype("int")  # approximate alignment
                    angle.append(steering_angle[idxs])
                    speed.append(speed_value[idxs])

                    goods = np.abs(angle[-1]) <= 200

                    filters.append(np.argwhere(goods)[time_len - 1:] + (lastidx + time_len - 1))
                    lastidx += goods.shape[0]
                    # check for mismatched length bug
                    print("x {} | t {} | f {}".format(x.shape[0], steering_angle.shape[0], goods.shape[0]))
                    if x.shape[0] != angle[-1].shape[0] or x.shape[0] != goods.shape[0]:
                        raise Exception("bad shape")

            except IOError:
                import traceback
                traceback.print_exc()
                print("failed to open", tword)

        angle = np.concatenate(angle, axis=0)
        speed = np.concatenate(speed, axis=0)
        filters = np.concatenate(filters, axis=0).ravel()
        print("training on %d/%d examples" % (filters.shape[0], angle.shape[0]))
        return c5x, angle, speed, filters, hdf5_camera

    def datagen(self, time_len=1, batch_size=256, ignore_goods=False):
        """
        Creates generatos for the datasets.
        Input:
        - datadir: path to the data directory
        - time_len: number of frames per data point
        - batch_size: data batch size
        - ignore_goods: Ignore `good` filters.
        Output:
        - Generator (X_batch, angle_batch, speed_batch) of size (batch_size, width, heigth, 3)
        """
        assert time_len > 0

        all_files = glob.glob(os.path.join(self.data_dir))
        filter_names = sorted(all_files)

        logger.info("Loading {} hdf5 buckets.".format(len(filter_names)))

        c5x, angle, speed, filters, hdf5_camera = self.concatenate(filter_names, time_len=time_len)
        filters_set = set(filters)

        logger.info("camera files {}".format(len(c5x)))

        X_batch = np.zeros((batch_size, time_len, 3, 160, 320), dtype='uint8')
        angle_batch = np.zeros((batch_size, time_len, 1), dtype='float32')
        speed_batch = np.zeros((batch_size, time_len, 1), dtype='float32')

        while True:
            try:
                t = time.time()

                count = 0
                start = time.time()
                while count < batch_size:
                    if not ignore_goods:
                        i = np.random.choice(filters)
                        # check the time history for goods
                        good = True
                        for j in (i - time_len + 1, i + 1):
                            if j not in filters_set:
                                good = False
                        if not good:
                            continue

                    else:
                        i = np.random.randint(time_len + 1, len(angle), 1)

                    # GET X_BATCH
                    # low quality loop
                    for es, ee, x in c5x:
                        if i >= es and i < ee:
                            X_batch[count] = x[i - es - time_len + 1:i - es + 1]
                            break

                    angle_batch[count] = np.copy(angle[i - time_len + 1:i + 1])[:, None]
                    speed_batch[count] = np.copy(speed[i - time_len + 1:i + 1])[:, None]

                    count += 1

                # sanity check
                assert X_batch.shape == (batch_size, time_len, 3, 160, 320)

                # logging.debug("loading image took: {}s".format(time.time()-t))
                # print("%5.2f ms" % ((time.time()-start)*1000.0))

                if self.first:
                    print("X", X_batch.shape)
                    print("angle", angle_batch.shape)
                    print("speed", speed_batch.shape)
                    self.first = False

                yield (X_batch, angle_batch, speed_batch)

            except KeyboardInterrupt:
                raise
            except GeneratorExit:
                return
            except:
                traceback.print_exc()
                pass

    def gen(self, time_len=1, batch_size=256, ignore_goods=False):
        """" Wrapper for datagen"""
        for data_row in self.datagen(time_len, batch_size, ignore_goods):
            X, angle, speed = data_row
            angle = angle[:, -1]
            speed = speed[:, -1]
            if X.shape[1] == 1:  # no temporal context
                X = X[:, -1]
            yield X, angle, speed

    def read_row(self):
        """ Reads one row of data
        :param data_dir: Data source path
        :output: One row of data
        """
        # Reading a batch of 1
        X, angle, speed = next(self.gen_line)
        angle = angle[:, -1]
        speed = speed[:, -1]
        if X.shape[1] == 1:  # no temporal context
            X = X[:, -1]
        return [X[0, :].astype(np.float32), angle[0, :], speed[0, :]]

    def read_row_tf(self):
        def fun():
            return self.read_row()

        return (tf.py_func(fun, [], [tf.float32, tf.float32, tf.float32]))


if __name__ == "__main__":
    # Testing only
    DATA_DIR = os.path.expanduser('~/Documents/data/comma/comma-additional/camera/training/2016-02-08--14-56-28.h5')
    reader = DataReader(DATA_DIR)
    x, y, s = reader.read_row_tf()

    with tf.Session() as sess:
        res = sess.run([x, y, s])
        print(">>>")
        print(res)
        res = sess.run([x, y, s])
        print(">>>2")
        print(res)
