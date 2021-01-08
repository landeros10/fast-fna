'''
Created on Sep 19, 2019
author: landeros10
'''
from __future__ import (print_function, division,
                        absolute_import, unicode_literals)
import numpy as np
from PIL import Image
from skimage.transform import rescale
import tensorflow as tf
import tensorflow_addons as tfa
import logging
from typing import List, Tuple


def make_weight_map(y_true: tf.Tensor,
                    n_class: int,
                    class_weights=None) -> tf.Tensor:
    """Create weight map using class_weights to scale loss by class.

    Args:
        class_weights (List): List of weights assigned to each class
        y_true (tf.Tensor): ground truth per-pixel (sparse) labels
        n_class (int): number of classes
    Returns:
        tf.Tensor: weight map tensor.
    """

    sample_weight = None
    if class_weights is not None:
        y_true = tf.one_hot(y_true, n_class, dtype=tf.float32)
        class_weights = tf.constant(np.array(class_weights),
                                    dtype=y_true.dtype)
        class_weights = tf.reshape(class_weights, (1, 1, 1, n_class))
        class_weights = class_weights / tf.reduce_sum(class_weights)

        sample_weight = tf.multiply(y_true, class_weights)
        sample_weight = tf.reduce_sum(sample_weight, axis=-1)

    return sample_weight


def make_batch_weights(y_true: tf.Tensor, p: float, limit: int = 200):
    """Make weight map that decreases importance of negative samples.
    Args:
        y_true (tf.Tensor): ground truth label
        p (float): scale to reduce importance of negative samples.
    """
    if p != 1.0:
        limit = tf.constant([limit], dtype=y_true.dtype)
        binary_y_true = tf.cast(y_true > 0, y_true.dtype)
        neg_ids = tf.reduce_sum(binary_y_true, axis=(1, 2)) < limit
        map = tf.where(neg_ids,
                       p * tf.ones_like(neg_ids, dtype=tf.float32),
                       tf.ones_like(neg_ids, dtype=tf.float32))
        return map / tf.reduce_sum(map)
    # return None if we do not want to weight by example
    return


def jaccard(y_true, logits, n_class, class_weights=None):
    """Jaccard loss function.

    Args:
        y_true (type): Sparse ground truth tensor, shape (N, H, W).
        logits (type): Logits tensor, shape (N, H, W, C).
        n_class (type): Num class, C.
        class_weights (type): Class weights, default None.

    Returns:
        type: tensorflow scalar.

    """
    prediction = tf.one_hot(tf.math.argmax(logits, axis=-1), n_class)
    y_true = tf.one_hot(y_true, n_class, dtype=prediction.dtype)

    eps = 1e-5
    intersect = tf.reduce_sum(prediction * y_true, axis=(1, 2))
    union = tf.reduce_sum(prediction + y_true, axis=(1, 2)) - intersect

    # Minimized when iou is 1
    iou_loss = 1 - ((intersect + eps) / (union + eps))

    if class_weights is not None:
        class_weights = tf.constant(np.array(class_weights),
                                    dtype=iou_loss.dtype)
        class_weights = tf.reshape(class_weights, (1, n_class))
        class_weights = class_weights / tf.reduce_sum(class_weights)
        iou_loss = tf.multiply(iou_loss, class_weights)

    return tf.math.reduce_sum(iou_loss, axis=1)


def to_rgb(img, amin=None, amax=None):
    """
    Converts the given array into a RGB image. If the number of channels is not
    3 the array is tiled such that it has 3 channels. Finally, the values are
    rescaled to [0,255)

    :param img: the array to convert [nx, ny, channels]

    :returns img: the rgb image [nx, ny, 3]
    """
    img = np.atleast_3d(img)
    channels = img.shape[2]
    if channels < 3:
        img = np.tile(img, 3)

    img[np.isnan(img)] = 0
    if amin is None:
        amin = np.amin(img)
    img -= amin

    if np.amax(img) != 0:
        if amax is None:
            amax = np.amax(img)
        img = img.astype(float) / amax

    img *= 255
    return img.astype(int)


def cropto(data, shape):
    """
    Crops the array to the given image shape by removing the border
    (expects a tensor of shape [batches, nx, ny, channels].
    Crops along dimensions 1 and 2.

    :param data: the array to crop
    :param shape: the target shape
    """
    diff_nx = (data.shape[1] - shape[1])
    diff_ny = (data.shape[2] - shape[2])

    offx_l = diff_nx // 2
    offx_r = diff_nx - offx_l
    offy_l = diff_ny // 2
    offy_r = diff_ny - offy_l

    cropped = data[:, offx_l:(-offx_r), offy_l:(-offy_r), ...]

    assert cropped.shape[1] == shape[1]
    assert cropped.shape[2] == shape[2]
    return cropped


def combine_preds(dataset_batch, pred, n_class, bs=25, resize_out=1.0):
    """
    Combines the data, grouth thruth and the prediction into one rgb image

    :param dataset_batch: x - (N, H, W, n_channel), y - (N, H, W),
    :param preds: predicted tensor size (bs, H, W)

    :returns img: the concatenated rgb image
    """
    ps = pred.shape
    width = ps[2]

    x, y = dataset_batch
    x, y = cropto(x, ps), cropto(y, ps)

    # convert to [bs x width] image
    x = x[:min(x.shape[0], bs), ...].numpy()
    input_signals = [to_rgb(x[..., ch].reshape(-1, width), amin=-1.0, amax=2.0)
                     for ch in range(x.shape[-1])]

    # convert to [bs x width] image
    y = y[:min(y.shape[0], bs), ...].numpy()
    pred = pred[:min(y.shape[0], bs), ...].numpy().astype(np.float32)

    gt = to_rgb(y.reshape(-1, width), amin=0, amax=n_class-1)
    p = to_rgb(pred.reshape(-1, width), amin=0, amax=n_class-1)

    # Combine and resize
    img = np.concatenate((*input_signals, gt, p), axis=1)
    img = rescale(img.astype(float), resize_out, multichannel=True).astype(int)
    return img


def save_image(img, path):
    """
    Writes the image to disk

    :param img: the rgb image to save
    :param path: the target path
    """
    im = Image.fromarray(img.round().astype(np.uint8))
    im.save(path, 'JPEG', dpi=[300, 300], quality=90)


def rotate(x: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """rotate x, y each with dim = 4"""
    # xdtype, ydtype = x.dtype, y.dtype
    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    x = tf.image.rot90(x, k)
    y = tf.image.rot90(y[..., np.newaxis], k)
    return x, y[..., 0]


def flip(x: tf.Tensor,
         y: tf.Tensor,
         p: float = 0.5) -> Tuple[tf.Tensor, tf.Tensor]:
    # xdtype, ydtype = x.dtype, y.dtype
    do_flip = tf.random.uniform([]) > p
    x = tf.cond(do_flip, lambda: tf.image.flip_left_right(x), lambda: x)
    y = tf.cond(do_flip, lambda: tf.image.flip_left_right(y), lambda: y)

    do_flip = tf.random.uniform([]) > p
    x = tf.cond(do_flip, lambda: tf.image.flip_up_down(x), lambda: x)
    y = tf.cond(do_flip, lambda: tf.image.flip_up_down(y), lambda: y)
    return x, y


def resize(x: tf.Tensor,
           y: tf.Tensor,
           resize_lims: List,
           p: float = 0.5) -> Tuple[tf.Tensor, tf.Tensor]:
    low, high = resize_lims
    if low == high:
        return x, y
    xdtype, ydtype = x.dtype, y.dtype
    bs = x.shape.as_list()[0]
    scales = np.random.uniform(low=low, high=high, size=[bs])
    boxes = np.zeros((bs, 4))

    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]

    resizex = tf.image.crop_and_resize(x,
                                       boxes=boxes,
                                       box_indices=np.arange(bs),
                                       crop_size=(x.shape[1], x.shape[2]),
                                       method='bilinear')
    resizey = tf.image.crop_and_resize(y[..., np.newaxis],
                                       boxes=boxes,
                                       box_indices=np.arange(bs),
                                       crop_size=(y.shape[1], y.shape[2]),
                                       method='bilinear')

    doCrop = tf.random.uniform([]) > p
    x = tf.cond(doCrop, lambda: x, lambda: tf.cast(resizex, xdtype))
    y = tf.cond(doCrop, lambda: y, lambda: tf.cast(resizey[..., 0], ydtype))
    return x, y


def add_noise(x: tf.Tensor, noiseDev: float, p: float = 0.5) -> tf.Tensor:
    doAdd = tf.random.uniform([]) < p
    noise = tf.random.normal(shape=x.shape, stddev=noiseDev, dtype=x.dtype)

    x = tf.cond(doAdd,
                lambda: x,
                lambda: tf.clip_by_value(tf.add(x, noise), -1.0, 1.0)
                )
    return x


def mean_shift(x: tf.Tensor, meanDev: float, p: float = 0.5) -> tf.Tensor:
    doShift = tf.random.uniform([]) < p
    noise = tf.random.uniform([], -meanDev, meanDev, dtype=x.dtype)

    x = tf.cond(doShift,
                lambda: x,
                lambda: tf.clip_by_value(tf.add(x, noise), -1.0, 1.0)
                )
    return x


def augment(x: tf.Tensor,
            y: tf.Tensor,
            resize_lims,
            noiseDev=1e-4,
            meanDev=1e-1,
            noise_p=0.25,
            mean_p=0.25,):
    """ Performs stochastic augmentaiton steps.

    Performs augmentation steps, using the following probability params:
    1. Rotation     .25 probability for (0, 90, 180, 270) degree rotation
    2. Flip         .5 probability of up/down and left/right flips, each.
    3. Resize       .5 probability of resize. parameterized by resize_lims
    4. Gaussian     .5 probability of gaussian noise addition.
    :param x (numpy array) array of N arrays, ith array size [1, H_i, W_i, C1]
    :param y (numpy array) array of N arrays, ith array size [1, H_i, W_i, C2]
    """
    x, y = rotate(x, y)
    x, y = flip(x, y)
    x, y = resize(x, y, resize_lims)
    x = add_noise(x, noiseDev, p=noise_p)
    x = mean_shift(x, meanDev, p=mean_p)
    return x, y


def affine_norm(input, mu, sigma):
    input_std = input.std(axis=(1, 2), keepdims=True)
    input_std[input_std == 0] = 1
    input_mean = input.mean(axis=(1, 2), keepdims=True)
    input -= input_mean
    input *= (sigma/input_std)
    return input + mu


def _parse_function(proto, shape):
    keys_to_features = {'x': tf.io.FixedLenFeature(shape, tf.float32),
                        'y': tf.io.FixedLenFeature(shape[:2], tf.int64)}
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)
    return parsed_features['x'], parsed_features['y']


def output_stats(acc_Object, loss, step):
    template = "Iter {}, Average Loss: {:.4f}, Accuracy: {:.4f}%"
    train_acc = acc_Object.result()*100
    logging.info(template.format(step, loss, train_acc))


def get_num_params(customModel):
    params = customModel.model.trainable_weights
    num_params = [np.prod(p.get_shape().as_list()) for p in params]
    return np.sum(num_params)
