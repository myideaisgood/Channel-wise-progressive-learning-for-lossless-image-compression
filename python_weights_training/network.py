import tensorflow as tf
import numpy as np
import os
import random
import cv2
import math

from config import parse_args
from module import model
from data import read_dir, write_tfrecord, read_tfrecord, data_exist

class Network(object):
    def __init__(self, args):
        self.args = args

        CKPT_DIR = args.ckpt_dir

        # Create ckpt directory if it does not exist
        if not os.path.exists(CKPT_DIR):
            os.mkdir(CKPT_DIR)

        # Print Arguments
        args = args.__dict__
        print("Arguments : ")

        for key, value in sorted(args.items()):
            print('\t%15s:\t%s' % (key, value))

        print("Are the arguments correct?")
        input("Press Enter to continue")

    def build(self):

        # Parameters
        DATA_DIR = self.args.data_dir

        LAYER_NUM = self.args.layer_num
        HIDDEN_UNIT = self.args.hidden_unit
        LAMBDA_CTX = self.args.lambda_ctx
        LAMBDA_Y = self.args.lambda_y
        LAMBDA_U = self.args.lambda_u
        LAMBDA_V = self.args.lambda_v
        LR = self.args.lr

        BATCH_SIZE = self.args.batch_size
        CROP_SIZE = self.args.crop_size

        CHANNEL_EPOCH = self.args.channel_epoch
        JOINT_EPOCH = self.args.joint_epoch

        tfrecord_name = 'train.tfrecord'

        if not data_exist(DATA_DIR, tfrecord_name):
            img_list = read_dir(DATA_DIR + 'train/')
            write_tfrecord(DATA_DIR, img_list, tfrecord_name)

        input_crop, _, _ = read_tfrecord(DATA_DIR, tfrecord_name, num_epochs=3*CHANNEL_EPOCH+JOINT_EPOCH,
                                        batch_size=4, min_after_dequeue=10, crop_size=CROP_SIZE)

        input_data, label = self.crop_to_data(input_crop)

        y_gt = tf.slice(label, [0, 0], [-1, 1])
        u_gt = tf.slice(label, [0, 1], [-1, 1])
        v_gt = tf.slice(label, [0, 2], [-1, 1])

        out_y, hidden_y = model(input_data, LAYER_NUM, HIDDEN_UNIT, 'pred_y')

        input_f2 = tf.concat([hidden_y, input_data, y_gt, tf.expand_dims(out_y[:,0], axis=1)], axis=1)

        out_u, hidden_u = model(input_f2, LAYER_NUM, HIDDEN_UNIT, 'pred_u')

        input_f3 = tf.concat([hidden_u, input_data, y_gt, tf.expand_dims(out_y[:, 0], axis=1), u_gt, tf.expand_dims(out_u[:, 0], axis=1)], axis=1)

        out_v, _, = model(input_f3, LAYER_NUM, HIDDEN_UNIT, 'pred_v')

        pred_y = out_y[:, 0]
        pred_u = out_u[:, 0]
        pred_v = out_v[:, 0]
        ctx_y  = tf.nn.relu(out_y[:, 1])
        ctx_u  = tf.nn.relu(out_u[:, 1])
        ctx_v  = tf.nn.relu(out_v[:, 1])

        predError_y = abs(tf.subtract(pred_y, tf.squeeze(y_gt, axis=1)))
        predError_u = abs(tf.subtract(pred_u, tf.squeeze(u_gt, axis=1)))
        predError_v = abs(tf.subtract(pred_v, tf.squeeze(v_gt, axis=1)))

        loss_pred_y = LAMBDA_Y * tf.reduce_mean(predError_y)
        loss_pred_u = LAMBDA_U * tf.reduce_mean(predError_u)
        loss_pred_v = LAMBDA_V * tf.reduce_mean(predError_v)

        loss_ctx_y = LAMBDA_Y * LAMBDA_CTX * tf.reduce_mean(abs(tf.subtract(ctx_y, predError_y)))
        loss_ctx_u = LAMBDA_U * LAMBDA_CTX * tf.reduce_mean(abs(tf.subtract(ctx_u, predError_u)))
        loss_ctx_v = LAMBDA_V * LAMBDA_CTX * tf.reduce_mean(abs(tf.subtract(ctx_v, predError_v)))

        loss_y = loss_pred_y + loss_ctx_y
        loss_u = loss_pred_u + loss_ctx_u
        loss_v = loss_pred_v + loss_ctx_v

        loss_yuv = loss_y + loss_u + loss_v

        t_vars = tf.trainable_variables()
        y_vars = [var for var in t_vars if 'pred_y' in var.name]
        u_vars = [var for var in t_vars if 'pred_u' in var.name]
        v_vars = [var for var in t_vars if 'pred_v' in var.name]

        self.optimizer_y = tf.train.AdamOptimizer(LR).minimize(loss_y, var_list=y_vars)
        self.optimizer_u = tf.train.AdamOptimizer(LR).minimize(loss_u, var_list=u_vars)
        self.optimizer_v = tf.train.AdamOptimizer(LR).minimize(loss_v, var_list=v_vars)
        self.optimizer_yuv = tf.train.AdamOptimizer(LR).minimize(loss_yuv, var_list=t_vars)

        # Variables
        self.loss_y = loss_y
        self.loss_u = loss_u
        self.loss_v = loss_v
        self.loss_yuv = loss_yuv
        self.loss_pred_y = loss_pred_y
        self.loss_pred_u = loss_pred_u
        self.loss_pred_v = loss_pred_v
        self.loss_pred_yuv = loss_pred_v + loss_pred_u + loss_pred_v
        self.loss_ctx_y = loss_ctx_y
        self.loss_ctx_u = loss_ctx_u
        self.loss_ctx_v = loss_ctx_v
        self.loss_ctx_yuv = loss_ctx_y + loss_ctx_u + loss_ctx_v
        self.ctx_y = ctx_y
        self.ctx_u = ctx_u
        self.ctx_v = ctx_v

    def crop_to_data(self, input_crop):

        # Parameters
        SUP_UP = self.args.sup_up
        SUP_LEFT = self.args.sup_left
        SUP_TOTAL = (SUP_LEFT * 2 + 1) * SUP_UP + SUP_LEFT
        BATCH_SIZE = self.args.batch_size
        CROP_SIZE = self.args.crop_size

        input_crop = tf.cast(input_crop, tf.float32)

        patch_size = int(CROP_SIZE * 2 / math.sqrt(BATCH_SIZE))

        # Obtain patch from cropped images
        random_patch = tf.extract_image_patches(input_crop, ksizes=[1, patch_size, patch_size, 1], strides=[1, patch_size, patch_size, 1], rates=[1,1,1,1], padding="VALID")
        random_patch = tf.reshape(random_patch, [-1, patch_size, patch_size, 3])
        data = tf.random_crop(random_patch, [BATCH_SIZE, SUP_UP+1, 2*SUP_LEFT+1, 3])
        data_reshape = tf.reshape(data, [BATCH_SIZE, -1, 3])

        # RGB2YUV
        data_rgb = data_reshape[:,:SUP_TOTAL+1,:]

        r,g,b = tf.split(data_rgb, 3, axis=2)

        u = b - tf.round((87 * r + 169 * g) / 256.0)
        v = r - g
        y = g + tf.round((86 * v + 29 * u) / 256.0)

        data_yuv = tf.concat([y,u,v], axis=2)
        data_yuv = tf.random.shuffle(data_yuv)

        # Subtract left pixel from support
        left_pixel = data_yuv[:,SUP_TOTAL-1,:]
        left_pixel = tf.expand_dims(left_pixel, axis=1)
        left_pixel = tf.tile(left_pixel, [1,SUP_TOTAL+1,1])

        data_yuv = data_yuv - left_pixel

        label = data_yuv[:,SUP_TOTAL,:]
        data_yuv = data_yuv[:,:SUP_TOTAL-1,:]

        data_y = data_yuv[:,:,0]
        data_u = data_yuv[:,:,1]
        data_v = data_yuv[:,:,2]

        input_data = tf.concat([data_y, data_u, data_v], axis=1)

        return input_data, label

    def train(self):

        GPU_NUM = self.args.gpu_num
        CKPT_DIR = self.args.ckpt_dir
        DATA_DIR = self.args.data_dir
        TENSORBOARD_DIR = self.args.tensorboard_dir
        LOAD = self.args.load
        CHANNEL_EPOCH = self.args.channel_epoch
        JOINT_EPOCH = self.args.joint_epoch
        BATCH_SIZE = self.args.batch_size
        PRINT_EVERY = self.args.print_every
        SAVE_EVERY = self.args.save_every

        # Assign GPU
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_NUM)

        global_step = tf.Variable(0, trainable=False)
        increase = tf.assign_add(global_step, 1)

        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord, start=True)

            saver = tf.train.Saver(max_to_keep=1)
            ckpt = tf.train.get_checkpoint_state(CKPT_DIR)
            writer = tf.summary.FileWriter(TENSORBOARD_DIR, graph=tf.get_default_graph())

            # Load model if trained before
            if ckpt and LOAD:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Model Loaded")

            # Load dataset
            epoch = sess.run(global_step)

            loss_pred_epoch_y = loss_pred_epoch_u = loss_pred_epoch_v = 0
            loss_ctx_epoch_y = loss_ctx_epoch_u = loss_ctx_epoch_v = 0

            while True:
                sess.run(increase)

                if epoch < CHANNEL_EPOCH :
                    if epoch == 0:
                        print("========== Train Y Channel ==========")
                    optimizer = self.optimizer_y
                elif epoch < 2*CHANNEL_EPOCH:
                    if epoch == CHANNEL_EPOCH:
                        print("========== Train U Channel ==========")
                    optimizer = self.optimizer_u
                elif epoch < 3*CHANNEL_EPOCH:
                    if epoch == 2*CHANNEL_EPOCH:
                        print("========== Train V Channel ==========")
                    optimizer = self.optimizer_v
                else:
                    if epoch == 3*CHANNEL_EPOCH:
                        print("========== Train YUV Channel ==========")
                    optimizer = self.optimizer_yuv

                _, loss_p_y, loss_p_u, loss_p_v, loss_c_y, loss_c_u, loss_c_v =\
                    sess.run([optimizer, self.loss_pred_y, self.loss_pred_u, self.loss_pred_v, 
                        self.loss_ctx_y, self.loss_ctx_u, self.loss_ctx_v])

                loss_pred_epoch_y += loss_p_y
                loss_pred_epoch_u += loss_p_u
                loss_pred_epoch_v += loss_p_v

                loss_ctx_epoch_y += loss_c_y
                loss_ctx_epoch_u += loss_c_u
                loss_ctx_epoch_v += loss_c_v

                if (epoch + 1) % PRINT_EVERY == 0:
                    
                    loss_pred_epoch_y /= PRINT_EVERY
                    loss_pred_epoch_u /= PRINT_EVERY
                    loss_pred_epoch_v /= PRINT_EVERY
                    loss_ctx_epoch_y /= PRINT_EVERY
                    loss_ctx_epoch_u /= PRINT_EVERY
                    loss_ctx_epoch_v /= PRINT_EVERY

                    loss_epoch_y = loss_pred_epoch_y + loss_ctx_epoch_y
                    loss_epoch_u = loss_pred_epoch_u + loss_ctx_epoch_u
                    loss_epoch_v = loss_pred_epoch_v + loss_ctx_epoch_v

                    print('%04d\n' % (epoch + 1),
                        '***Y***   lossPred=', '{:9.4f}'.format(loss_pred_epoch_y), 'lossContext=', '{:9.4f}'.format(loss_ctx_epoch_y), 'Loss=', '{:9.4f}\n'.format(loss_epoch_y),
                        '***U***   lossPred=', '{:9.4f}'.format(loss_pred_epoch_u), 'lossContext=', '{:9.4f}'.format(loss_ctx_epoch_u), 'Loss=', '{:9.4f}\n'.format(loss_epoch_u),
                        '***V***   lossPred=', '{:9.4f}'.format(loss_pred_epoch_v), 'lossContext=', '{:9.4f}'.format(loss_ctx_epoch_v), 'Loss=', '{:9.4f}\n'.format(loss_epoch_v),
                        '***YUV*** lossPred=', '{:9.4f}'.format(loss_pred_epoch_y + loss_pred_epoch_u + loss_pred_epoch_v), 'lossContext=',
                        '{:9.4f}'.format(loss_ctx_epoch_y + loss_ctx_epoch_u + loss_ctx_epoch_v), 'Loss=', '{:9.4f}'.format(loss_epoch_y + loss_epoch_u + loss_epoch_v))

                    loss_pred_epoch_y = loss_pred_epoch_u = loss_pred_epoch_v = 0
                    loss_ctx_epoch_y = loss_ctx_epoch_u = loss_ctx_epoch_v = 0

                if (epoch + 1) % SAVE_EVERY == 0:
                    saver.save(sess, CKPT_DIR + 'model_', global_step=epoch + 1)
                    self.print_weights('y')
                    self.print_weights('u')
                    self.print_weights('v')
                    print("Model Saved")

                epoch = sess.run(global_step)

            coord.request_stop()
            coord.join(threads)

    def print_weights(self, channel='y'):

        HIDDEN_UNIT = self.args.hidden_unit
        SUP_UP = self.args.sup_up
        SUP_LEFT = self.args.sup_left

        W = [v for v in tf.trainable_variables() if (('kernel' in v.name) and ('pred_' + channel in v.name))]
        b = [v for v in tf.trainable_variables() if (('bias' in v.name) and ('pred_' + channel in v.name))]

        n_layer = len(W)

        W_ = []
        b_ = []

        for i in range(n_layer):
            W_.append(W[i].eval())
            b_.append(b[i].eval())

        n_in = W_[0].shape[0]
        n_hidden = HIDDEN_UNIT
        n_out = W[-1].shape[1]

        filename = 'weights_' + channel + '.txt'

        f = open(filename, 'w')

        f.write(str(n_in) + '\n')
        f.write(str(n_hidden) + '\n')
        f.write(str(n_out) + '\n')
        f.write(str(n_layer) + '\n')
        f.write(str(SUP_UP) + '\n')
        f.write(str(SUP_LEFT) + '\n')

        for k in range(n_layer):
            for i in range(W_[k].shape[0]):
                for j in range(W_[k].shape[1]):
                    f.write(str(W_[k][i,j]) + '\t')
                f.write('\n')

        for k in range(n_layer):
            for j in range(b_[k].shape[0]):
                f.write(str(b_[k][j]) + '\t')
            f.write('\n')


        f.close()

        filename = os.path.abspath("../") + "/c_compression/x64/Release/weights_" + channel + '.txt'

        f = open(filename, 'w')

        f.write(str(n_in) + '\n')
        f.write(str(n_hidden) + '\n')
        f.write(str(n_out) + '\n')
        f.write(str(n_layer) + '\n')
        f.write(str(SUP_UP) + '\n')
        f.write(str(SUP_LEFT) + '\n')

        for k in range(n_layer):
            for i in range(W_[k].shape[0]):
                for j in range(W_[k].shape[1]):
                    f.write(str(W_[k][i,j]) + '\t')
                f.write('\n')

        for k in range(n_layer):
            for j in range(b_[k].shape[0]):
                f.write(str(b_[k][j]) + '\t')
            f.write('\n')


        f.close()

    def print_all_weights(self):

        GPU_NUM = self.args.gpu_num
        CKPT_DIR = self.args.ckpt_dir

        # Assign GPU
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_NUM)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=1)
            ckpt = tf.train.get_checkpoint_state(CKPT_DIR)

            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Model Loaded")

            self.print_weights('y')
            self.print_weights('u')
            self.print_weights('v')
                    

if __name__ == "__main__":

    args = parse_args()
    my_net = Network(args)
    my_net.build()
    my_net.train()
    my_net.print_all_weights()
