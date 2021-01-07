'''
Created on Sep 19, 2019
author: landeros10
'''
from __future__ import (print_function, division,
                        absolute_import, unicode_literals)
from collections import OrderedDict
import pathlib
import logging
from datetime import datetime
import time
from os.path import join, abspath, exists
from os import makedirs
import abc
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras import optimizers
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.losses import SparseCategoricalCrossentropy, Reduction
from tensorflow.keras.mixed_precision import experimental as mp
from tensorflow.compat.v2.train import CheckpointManager

from unet_inj.layers import (prelu, relu, tanh, conv2d, deconv2d,
                             linear, avg_pool, max_pool,
                             crop_concat, dapi_add, dapi_process, dropout_bn)
from unet_inj.util import (make_weight_map, make_batch_weights, jaccard,
                           cropto, combine_preds, save_image,
                           augment, output_stats, _parse_function, get_count)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class Custom_Model():
    __metaclass__ = abc.ABCMeta

    def __init__(self,
                 channels: int,
                 n_outputs: int,
                 n_class: int,
                 input_shape: Tuple,
                 net_kwargs: Dict,
                 policy: str = "mixed_float16"):
        self.n_channels = channels
        self.n_out = n_outputs
        self.n_class = n_class
        self.net_kwargs = net_kwargs
        self.strategy = tf.distribute.MirroredStrategy()
        self.input_shape = input_shape
        self.build_model(policy)

    def build_model(self, policy):
        if policy == "mixed_float16":
            self.policy = mp.Policy(policy)
            mp.set_policy(self.policy)
        with self.strategy.scope():
            self.x = Input(shape=self.input_shape, name="x", dtype="float32")
            logits = self.define_model(self.x)
            self.model = Model(inputs=self.x, outputs=logits)
            self.define_loss()

    # Must be implemented in any Custom_Model object
    @abc.abstractmethod
    def define_model(self, input: tf.Tensor) -> tf.Tensor:
        """Take input tensor, return logits tensor."""

    @abc.abstractmethod
    def define_loss(self):
        """Store loss and accuracy functions. """
        # Any funcs needed by self.eval_loss
        self.loss = lambda input, y_true, y_pred: None

        # Must be defined
        self.tst_loss = None
        self.trn_acc = None
        self.tst_acc = None

    @abc.abstractmethod
    def eval_loss(self,
                  input: tf.Tensor,
                  y_true: tf.Tensor,
                  y_pred: tf.Tensor) -> tf.Tensor:
        """Uses loss funcs defined in self.define_loss to return loss"""
        return self.loss(input, y_true, y_pred)

    def gen_checkpoint(self, optimizer):
        self.checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                              model=self.model)

        self.ckpt_manager = CheckpointManager(self.checkpoint,
                                              directory=self.ckpts,
                                              max_to_keep=5)

    def restore(self, model_path):
        """
        Restores a session from a checkpoint
        :param model_path: path to file system checkpoint location
        """
        if not hasattr(self, "checkpoint"):
            self.ckpts = join(model_path, "saved_model")
            self.checkpoint = tf.train.Checkpoint(model=self.model)
            self.ckpt_manager = CheckpointManager(self.checkpoint,
                                                  directory=self.ckpts,
                                                  max_to_keep=5)

        self.checkpoint.restore(self.ckpt_manager.latest_checkpoint)
        logging.info("Model restored from file: %s" % model_path)

    def summary(self):
        self.model.summary()

    @abc.abstractmethod
    def store_prediction(self, data, name: str):
        """Creates predictions for a test batch. Saves images to file."""

    def parse_train_params(self):
        """ Parse through training parameters. Descriptions below:

            "bs"                e
            "learning_rate"     e
            "trn_data_len"      r
            "trn_data_len"      e
            "preFetch"          e
            "epochs"            tre
            "resize_lims"       [0.9, 1.0]
            "noiseDev"          1e-4
            "l2_norm"           0.1
            "display_step"      1
            "store_every"      e
            "store_size"        number of samples to store in prediction images
            "store_resize_f"    factor for resizing prediction images
            "num_losses"        e
            "loss_names"        e
            "bestAcc"           e
        """
        self.bs = self.train_kwargs.get("bs", 16)
        self.lr = self.train_kwargs.get("learning_rate", 3e-6)
        with self.strategy.scope():
            self.optimizer_type = self.train_kwargs.get("optimizer", "Adam")
            self.optimizer = optimizers.Adam(lr=self.lr)
            if self.optimizer_type == "RMSprop":
                print("Using RMSProp\n\n\n")
                self.optimizer = optimizers.RMSprop(learning_rate=self.lr)
            elif self.optimizer_type == "Adadelta":
                print("Using Adadelta \n\n\n")
                self.optimizer = optimizers.Adadelta(learning_rate=self.lr)
            elif self.optimizer_type == "Nadam":
                print("Using Nadam \n\n\n")
                self.optimizer = optimizers.Nadam(learning_rate=self.lr)
            self.optimizer = mp.LossScaleOptimizer(self.optimizer,
                                                   loss_scale='dynamic')

        # Can be used in initial dataset preparation
        self.trn_data_len = self.train_kwargs.get("trn_data_len", 100)
        self.tst_data_len = self.train_kwargs.get("tst_data_len", 100)
        self.preFetch = self.train_kwargs.get("preFetch",
                                              tf.data.experimental.AUTOTUNE)

        # Used in self.train_on_dataset
        self.epochs = self.train_kwargs.get("epochs", 100)
        #    Data augment parameters
        self.rsl = self.train_kwargs.get("resize_lims", [0.9, 1.0])
        self.noise = self.train_kwargs.get("noiseDev", 1e-4)
        self.meanDev = self.train_kwargs.get("meanDev", 1e-1)
        self.l2_coeff = self.train_kwargs.get("l2_norm", 0.0)
        self.noise_p = self.train_kwargs.get("noise_p", 0.25)
        self.mean_p = self.train_kwargs.get("mean_p", 0.25)

        #   Train processs logging and prediction storage params
        self.display_step = self.train_kwargs.get("display_step", 1)
        self.store_every = self.train_kwargs.get("store_every",
                                                 self.epochs//100)
        self.store_size = self.train_kwargs.get("store_size", 5)
        self.store_resize_f = self.train_kwargs.get("store_resize_f", 1.0)
        self.num_losses = self.train_kwargs.get("num_losses", 1)
        self.loss_names = self.train_kwargs.get("loss_names", None)
        #   Accuracy used in saving models
        self.bestAcc = self.train_kwargs.get("bestAcc", 0)

    def store_kwargs(self, train_kwargs={}, loss_kwargs={}):
        self.train_kwargs = train_kwargs
        self.parse_train_params()
        self.loss_kwargs = loss_kwargs

        # Paths to save files for documenting traing progress
        self.pred_path = self.train_kwargs.get("pred_path", "./prediction")
        self.out_path = self.train_kwargs.get("out_path", "./outputs")
        self.log_path = self.train_kwargs.get("log_path", "./logs")
        self.ckpts = join(self.out_path, "saved_model")
        log = "log-" + datetime.today().strftime("%y-%b-%d-%H") + ".txt"
        self.logFile = join(self.log_path, log)

        if not exists(abspath(self.log_path)):
            logging.info("Allocating '{:}'".format(self.log_path))
            makedirs(abspath(self.log_path))

        if not exists(abspath(self.pred_path)):
            logging.info("Allocating '{:}'".format(self.pred_path))
            makedirs(abspath(self.pred_path))

        if not exists(abspath(self.out_path)):
            logging.info("Allocating '{:}'".format(self.out_path))
            makedirs(abspath(self.out_path))

        # Start log file
        template = ("Optimizer: {}, LR {:.2e}, BS: {}\n"
                    "autoencoder: {:.2e}, count_f: {:.2e},\n"
                    "cce_f: {:.2e}, dice_f: {:.2e},\n"
                    "class weights: {} \n\n")
        template = template.format(self.optimizer_type,
                                   self.lr,
                                   self.bs,
                                   self.loss_kwargs.get("auto_coeff", 0.5),
                                   self.loss_kwargs.get("count_coeff", 1.0),
                                   self.loss_kwargs.get("cce_coeff", 1.0),
                                   self.loss_kwargs.get("dice_coeff", 100),
                                   self.loss_kwargs.get("class_weights",
                                                        [1, 1, 1])
                                   )
        f = open(self.logFile, "w")
        f.write("ISL training\n")
        f.write(template)

        trnKeys = ['trn_data_len', 'epochs', 'display_step', 'store_epochs',
                   'bestAcc', 'bs', 'learning_rate']
        trnDefaultVals = [10, 20, 1, 5, 0, 16, 3e-6]
        trnVals = [self.train_kwargs.get(trnKeys[i], trnDefaultVals[i])
                   for i in range(len(trnKeys))]
        template = ("Data Size {}, Num epochs {}, display every {},"
                    "store every {} Best Acc.: {}, Batch Size {}, LR: {}\n")
        f.write(template.format(*trnVals))
        f.close()

    # Distributed implementation, uses all GPUs available
    def train_on_dataset(self,
                         trn: tf.data.Dataset,
                         tst: tf.data.Dataset,
                         restore: bool):
        """Trains model on Dataset object using parameters in self.train_kwargs

        Args:
            trn (tf.data.Dataset): Training Dataset. Yields (in, out) tuples.
            tst (tf.data.Dataset): Training Dataset. Yields (in, out) tuples.
            restore (bool): Restore previously trained model.
                Must provide "out_path" in train_kwargs
        """
        if self.epochs == 0:
            return self.ckpts

        with self.strategy.scope():
            self.gen_checkpoint(self.optimizer)
            if restore:
                self.restore(self.out_path)

            def train_step(inputs):
                images, labels = inputs
                images, labels = augment(images, labels, self.rsl,
                                         noiseDev=self.noise,
                                         meanDev=self.meanDev,
                                         noise_p=self.noise_p,
                                         mean_p=self.mean_p)
                with tf.GradientTape() as gt:
                    pred = self.model(images, training=True)
                    if isinstance(pred, list):
                        pred_shape = pred[-1].shape
                        x = cropto(tf.split(images, 2, axis=3)[0], pred_shape)
                    else:
                        pred_shape = pred.shape
                    labels = cropto(labels, pred_shape)
                    loss = tf.reduce_sum(self.eval_loss(labels, pred, x)[0])
                    sc_loss = self.optimizer.get_scaled_loss(loss)
                sc_grads = gt.gradient(sc_loss, self.model.trainable_variables)
                grads = self.optimizer.get_unscaled_gradients(sc_grads)
                gradient_pairs = zip(grads, self.model.trainable_variables)
                self.optimizer.apply_gradients(gradient_pairs)
                if isinstance(pred, list):
                    pred = pred[-1]
                self.trn_acc.update_state(labels, pred)
                return loss

            def test_step(inputs):
                images, labels = inputs
                pred = self.model(images, training=False)
                if isinstance(pred, list):
                    pred_shape = pred[-1].shape
                    x = tf.split(images, 2, axis=3)[0]
                else:
                    pred_shape = pred.shape
                labels = cropto(labels, pred_shape)
                x = cropto(x, pred_shape)
                t_loss, _ = self.eval_loss(labels, pred, x)
                self.tst_loss.update_state(tf.reduce_sum(t_loss))
                if isinstance(pred, list):
                    pred = pred[-1]
                self.tst_acc.update_state(labels, pred)
                return t_loss

            @tf.function
            def dist_trn_step(data):
                rep_loss = self.strategy.experimental_run_v2(train_step,
                                                             args=(data,))
                loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM,
                                            rep_loss, axis=None)
                return loss

            @tf.function
            def dist_tst_step(data):
                rep_loss = self.strategy.experimental_run_v2(test_step,
                                                             args=(data,))
                loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM,
                                            rep_loss, axis=None)
                return loss

        for layer in self.model.layers:
            if (isinstance(layer, tf.keras.layers.Conv2D) or
                    isinstance(layer, tf.keras.layers.Dense)):
                l2_reg = tf.keras.regularizers.l2(self.l2_coeff)
                layer.add_loss(lambda: l2_reg(layer.kernel))

        logging.info("Start Optimization")
        startTime = time.time()
        with self.strategy.scope():
            trn_dist = self.strategy.experimental_distribute_dataset(trn)
            tst_dist = self.strategy.experimental_distribute_dataset(tst)
        for epoch in range(self.epochs):
            logging.info("Epoch {}".format(epoch))
            total_loss = 0
            step = 0
            with self.strategy.scope():
                for batch in trn_dist:
                    loss = dist_trn_step(batch)
                    total_loss += loss
                    if step % self.display_step == 0:
                        output_stats(self.trn_acc, loss, step)
                    step += 1
                    if step >= self.trn_data_len // self.bs - 1:
                        break
            train_loss = total_loss / step

            tst_loss = np.zeros((self.num_losses,))
            step = 0
            with self.strategy.scope():
                for batch in tst_dist:
                    t = dist_tst_step(batch).numpy()
                    tst_loss += t
                    step += 1
                    if step >= self.tst_data_len // self.bs - 1:
                        break
            train_acc = self.trn_acc.result() * 100
            test_loss = np.sum(tst_loss) / step
            test_acc = self.tst_acc.result() * 100

            if epoch % self.store_every == 0:
                self.store_prediction(next(iter(tst.take(1))),
                                      "epoch_{:03d}".format(epoch+1),
                                      size=self.store_size)
            evalTime = (time.time() - startTime) / 60

            sub_temp = ""
            if self.num_losses > 1:
                losspairs = zip(self.loss_names, tst_loss / step)
                sub_temp = ''.join('%s: %.3f ' % pair for pair in losspairs)

            temp = "".join(("Epoch {}, Time:{:.1f}m, Loss: {:.3f}, ",
                            "Accuracy: {:.3f}, ", "Test Loss: {:.3f},  ",
                            sub_temp, "Test Accuracy: {:.3f}"))
            temp_format = temp.format(epoch+1, evalTime, train_loss, train_acc,
                                      test_loss, test_acc)
            logging.info(temp_format)
            f = open(self.logFile, "a")
            f.write(temp_format)
            f.close()

            if test_acc > self.bestAcc:
                with self.strategy.scope():
                    self.ckpt_manager.save()
                logging.info("Saved Model!")

                f = open(self.logFile, "a")
                f.write("\t\tSaved Model!")
                f.close()

                self.bestAcc = test_acc
            f = open(self.logFile, "a")
            f.write("\n")
            f.close()

            with self.strategy.scope():
                train_acc = self.trn_acc.result() * 100
                test_loss = self.tst_loss.result()
                test_acc = self.tst_acc.result() * 100

        logging.info("Optimization Finished!")

    # Must implement train function in any new model class
    # self.train_on_dataset provided, so self.train needs to prepare
    # training and testing dataset objects (from array, from file, etc.)
    @abc.abstractmethod
    def train(self, data, restore=False, train_kwargs={}, loss_kwargs={}):
        """ Begin training process. First stores/parses through kwargs """
        self.store_kwargs(train_kwargs, loss_kwargs)
        self.train_on_dataset(data[0], data[1], restore=restore)


class Unet_Inj(Custom_Model):

    def __init__(self,
                 n_class: int = 3,
                 n_outputs: int = 1,
                 net_kwargs: Dict = {},
                 policy: str = "mixed_float16"):
        n_channels = 2
        self.input_s = net_kwargs.get("input_s", None)
        input_shape = (None, None, n_channels)

        super().__init__(n_channels,
                         n_outputs,
                         n_class,
                         input_shape,
                         net_kwargs,
                         policy=policy)

    def define_model(self, input: tf.Tensor) -> tf.Tensor:
        """Take input tensor, return logits tensor using Unet_Inj architecture.
        Args:
            input (tf.Tensor): input tensor. Shape (n, s, s, n_channel)
        Returns:
            tf.Tensor: logit tensor. Shape (n, s-off, s-off, n_class).
                where off represents a cutoff due to valid padding effects.
        """

        # Split into max projection and dapi images
        in_node, dapi = tf.split(input, 2, axis=3)

        # Encoding architecture parameters
        layers = self.net_kwargs.get("layers", 3)
        feat_factor = self.net_kwargs.get("feat_factor", 2)
        features_root = self.net_kwargs.get("features_root", 16)
        filter_size = self.net_kwargs.get("filter_size", 3)
        rate = self.net_kwargs.get("rate", 0.1)
        pool_size = self.net_kwargs.get("pool_size", 2)

        # Encoding process
        dw_h_convs = OrderedDict()
        for layer in range(0, layers):
            with tf.name_scope("down_conv_{}".format(str(layer))):
                features = int((feat_factor ** layer) * features_root)
                conv1 = prelu(conv2d(in_node, features, filter_size))
                conv1 = dropout_bn(conv1, rate)
                conv2 = prelu(conv2d(conv1, features, filter_size))
                conv2 = dropout_bn(conv2, rate)
                dw_h_convs[layer] = conv2
                if layer < layers - 1:
                    in_node = max_pool(dw_h_convs[layer], pool_size)
        in_node = dw_h_convs[layers - 1]

        # Parameters for Dapi signal introduction
        dapiFeat = self.net_kwargs.get("dapiFeat", 8)
        dapi_position = self.net_kwargs.get("dapi_position", "last")
        joinType = self.net_kwargs.get("joinType", "concat")

        # Decoding process, including dapi signal addition
        for layer in range(layers - 2, -1, -1):
            with tf.name_scope("up_conv_{}".format(str(layer))):
                # Deconvolve and concatenate to stored layer
                features = int(feat_factor ** (layer + 1) * features_root)
                deconv1 = prelu(deconv2d(in_node, features, pool_size))
                deconv1 = dropout_bn(deconv1, rate)
                f = int((feat_factor ** layer) * features_root)
                deconv2 = crop_concat(dw_h_convs[layer], deconv1, f)

                conv1 = prelu(conv2d(deconv2, features // 2, filter_size))
                conv1 = dropout_bn(conv1, rate)
                if layer == 0 and dapi_position == "second":
                    dapi = dapi_process(dapi, dapiFeat, filter_size, rate)
                    conv1 = dapi_add(dapi, conv1, joinType, dapiFeat)

                conv2 = prelu(conv2d(conv1, features // 2, filter_size))
                conv2 = dropout_bn(conv2, rate)
                if layer == 0 and dapi_position == "last":
                    dapi = dapi_process(dapi, dapiFeat, filter_size, rate)
                    conv2 = dapi_add(dapi, conv2, joinType, dapiFeat)
                in_node = conv2

        # Output Map
        with tf.name_scope("output_map"):
            logits = conv2d(in_node, self.n_class, 1, dtype=tf.float32)
        return logits

    def define_loss(self):
        """Store attributes containing loss and accuracy functions.

        Training loss is defined by Cross Categorical Cross Entropy and by
        Dice loss, an intersection-over-union loss function.
        """
        cce = SparseCategoricalCrossentropy(from_logits=True,
                                            reduction=Reduction.NONE,
                                            name='sparse_crossentropy')

        self.cce_loss = cce
        self.dice_loss = jaccard

        self.tst_loss = tf.keras.metrics.Mean(name='test_loss')
        self.trn_acc = SparseCategoricalAccuracy(name='trn_acc')
        self.tst_acc = SparseCategoricalAccuracy(name='tst_acc')

    def eval_loss(self,
                  y_true: tf.Tensor,
                  y_pred: tf.Tensor,
                  x: tf.Tensor) -> tf.Tensor:
        """Evaluates loss for a given set of inputs and prediction.

        Uses loss functions defined in self.define_loss.
        Class weights defined by parameter class_weights in self.loss_kwargs
        Dice loss is scaled by parameter dice_coeff in self.loss_kwargs

        Args:
            input (tf.Tensor): Description of parameter `input`.
            y_true (tf.Tensor): Description of parameter `y_true`.
            y_pred (tf.Tensor): Description of parameter `y_pred`.

        Returns:
            tf.Tensor: Description of returned object.

        """
        class_weights = self.loss_kwargs.get("class_weights", None)
        sample_weight = make_weight_map(y_true, self.n_class,
                                        class_weights=class_weights)
        loss_cce = self.cce_loss(y_true, y_pred, sample_weight=sample_weight)
        loss_cce = tf.math.reduce_sum(loss_cce, axis=[1, 2])

        dice_coeff = self.loss_kwargs.get("dice_coeff", 100)
        loss_iou = self.dice_loss(y_true, y_pred, self.n_class, class_weights)

        loss = loss_cce + dice_coeff * (loss_iou)

        batch_weight_f = self.loss_kwargs.get("batch_weight_f", 1.0)
        pos_limit = self.loss_kwargs.get("limit", 200)
        batch_weights = make_batch_weights(y_true, batch_weight_f, pos_limit)
        loss = tf.nn.compute_average_loss(loss,
                                          sample_weight=batch_weights,
                                          global_batch_size=self.bs)
        return loss, sample_weight

    def store_prediction(self, tst_batch, name, size=5):
        pred = self.model(tst_batch, training=False)
        pred = tf.math.argmax(pred, axis=-1)
        img = combine_preds(tst_batch, pred, self.n_class,
                            bs=size,
                            resize_out=self.store_resize_f)
        save_image(img, "%s/%s.png" % (self.pred_path, name))

    def train(self, trn, tst, restore=False, train_kwargs={}, loss_kwargs={}):
        n_trn = len(list(pathlib.Path(join(trn, "tiles")).glob("*_x_*.npy")))
        logging.info("Training File Path : {}".format(join(trn, "tiles")))
        logging.info("Training Samples: {}".format(n_trn))
        train_kwargs["trn_data_len"] = n_trn

        n_tst = len(list(pathlib.Path(join(tst, "tiles")).glob("*_x_*.npy")))
        logging.info("Testing File Path : {}".format(join(tst, "tiles")))
        logging.info("Testing Samples: {}\n\n".format(n_tst))
        train_kwargs["tst_data_len"] = n_tst

        self.store_kwargs(train_kwargs, loss_kwargs)

        shape = (self.input_s, self.input_s, self.n_channels)

        trn = tf.data.TFRecordDataset(filenames=join(trn, "data.tfrecord"))
        trn = trn.shuffle(self.trn_data_len//4).repeat()
        trn = trn.map(lambda s: _parse_function(s, shape))
        trn = trn.batch(self.bs)
        trn = trn.prefetch(buffer_size=self.preFetch)

        tst = tf.data.TFRecordDataset(filenames=join(tst, "data.tfrecord"))
        tst = tst.shuffle(self.tst_data_len//4).repeat()
        tst = tst.map(lambda s: _parse_function(s, shape))
        tst = tst.batch(self.bs)
        tst = tst.prefetch(buffer_size=self.preFetch)

        self.train_on_dataset(trn, tst, restore=restore)
