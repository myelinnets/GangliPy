u = "http://unifoundry.com/pub/unifont/unifont-12.0.01/unifont-12.0.01.bmp"

import requests
import os
from PIL import Image
import numpy as np
from io import BytesIO
import logging
from cvpubsubs.webcam_pub import VideoHandlerThread, display_callbacks
from cvpubsubs.window_sub import SubscriberWindows
import time
from random import shuffle


def check_downloaded():
    return os.path.isfile('unifoundry.npz')


def download():
    if check_downloaded():
        logging.info("Unifoundry dataset already exists at unifoundry.npz. Exiting.")
        return

    file = requests.get(u)
    im = Image.open(BytesIO(file.content))
    p = np.array(im)
    p_chars = p[64:, 32:]

    dict = {}
    bx = np.split(p_chars, 0x100, axis=0)
    for i in range(len(bx)):
        by = np.split(bx[i], 0x100, axis=1)
        for j in range(len(by)):
            b_arr = by[j]
            dict.update({"chr_{}.npy".format(i * 0x100 + j): b_arr})

    np.savez_compressed('unifoundry.npz', **dict)


def get_dataset():
    download()

    np_set = np.load('unifoundry.npz')

    return np_set


def shuffled_iter():
    np_set = get_dataset()

    while True:
        shuffle(np_set.files)
        for np_file in np_set.files:
            yield np_set[np_file]


def shuffled_batch_iter(batch_size):
    it = shuffled_iter()
    while True:
        arr = np.empty((batch_size, 16, 16, 1), dtype=bool)
        for b in range(batch_size):
            arr[b, ...] = next(it)[..., np.newaxis]
        yield arr


def display():
    img_gen = shuffled_batch_iter(64)

    arr = np.empty((1, 16 * 8, 16 * 8, 1))

    def img_loop(frame, id):
        ar = next(img_gen)
        ar1 = []
        for i in range(8):
            ar1.append(np.concatenate([ar[x] for x in range(i * 8, i * 8 + 8)], axis=0))
        ar2 = np.concatenate(ar1, axis=1)

        arr[:] = ar2
        return arr

    v = VideoHandlerThread(video_source=arr, callbacks=display_callbacks + [img_loop])
    s = SubscriberWindows(video_sources=[arr], window_names=[str(x) for x in range(1)])

    v.start()
    s.loop()
    v.join()


if __name__ == "__main__":
    display()
