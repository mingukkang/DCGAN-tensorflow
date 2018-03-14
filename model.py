import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os

class Gan():

    def __init__(self,sess,data_shape,z_d, learning_rate,batch_size):
        self.sess  = sess
        self.data_shape = data_shape
        self.length = self.data_shape[0]*self.data_shape[1]*self.data_shape[2]
        self.z_d = z_d
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.beta1 = 0.5
        self.build_net()

    def Generator(self, z, is_training, reuse):
        depths = [1024, 512, 256, 128] + [self.data_shape[2]]
        with tf.variable_scope("Generator", reuse = reuse):
            with tf.variable_scope("g_fc1", reuse = reuse):
                output = tf.layers.dense(z, depths[0]*4*4, trainable = is_training)
                output = tf.reshape(output, [self.batch_size, 4, 4, depths[0]])
                output = tf.nn.relu(tf.layers.batch_normalization(output, training = is_training))

            with tf.variable_scope("g_dc1", reuse = reuse):
                output = tf.layers.conv2d_transpose(output, depths[1], [5,5], strides =(2,2), padding ="SAME", trainable = is_training)
                output = tf.nn.relu(tf.layers.batch_normalization(output, training = is_training))

            with tf.variable_scope("g_dc2", reuse = reuse):
                output = tf.layers.conv2d_transpose(output, depths[2], [5,5], strides = (2,2), padding ="SAME", trainable = is_training)
                output = tf.nn.relu(tf.layers.batch_normalization(output, training = is_training))

            with tf.variable_scope("g_dc3", reuse = reuse):
                output = tf.layers.conv2d_transpose(output,depths[3], [5,5], strides = (2,2), padding ="SAME", trainable = is_training)
                output = tf.nn.relu(tf.layers.batch_normalization(output, training = is_training))

            with tf.variable_scope("g_dc4", reuse = reuse):
                output = tf.layers.conv2d_transpose(output,depths[4], [5,5], strides = (2,2), padding = "SAME", trainable = is_training)
                g_logits = tf.nn.tanh(output)

        return g_logits

    def Discriminator(self,X, is_training, reuse):
        depths = [self.data_shape[2]] + [64, 128, 256, 512]
        with tf.variable_scope("Discriminator", reuse = reuse):
            with tf.variable_scope("d_cv1", reuse = reuse):
                output = tf.layers.conv2d(X, depths[1], [5,5], strides = (2,2), padding ="SAME", trainable = is_training)
                output = tf.nn.leaky_relu(tf.layers.batch_normalization(output, training = is_training))

            with tf.variable_scope("d_cv2", reuse = reuse):
                output = tf.layers.conv2d(output, depths[2], [5,5], strides = (2,2), padding ="SAME", trainable = is_training)
                output = tf.nn.leaky_relu(tf.layers.batch_normalization(output, training = is_training))

            with tf.variable_scope("d_cv3", reuse = reuse):
                output = tf.layers.conv2d(output, depths[3], [5,5], strides = (2,2), padding = "SAME", trainable = is_training)
                output = tf.nn.leaky_relu(tf.layers.batch_normalization(output, training = is_training))

            with tf.variable_scope("d_cv4", reuse = reuse):
                output = tf.layers.conv2d(output, depths[4], [5,5], strides = (2,2), padding ="SAME", trainable = is_training)
                output = tf.nn.leaky_relu(tf.layers.batch_normalization(output, training = is_training))

            with tf.variable_scope("d_fc1", reuse = reuse):
                output = tf.layers.flatten(output)
                d_logits = tf.layers.dense(output,1, trainable = is_training)

            return d_logits

    def plot_and_save(self, order, images):
        batch_size = len(images)
        n = np.int(np.sqrt(batch_size))
        image_size = np.shape(images)[2]
        n_channel = np.shape(images)[3]
        images = np.reshape(images, [-1,image_size,image_size,n_channel])
        canvas = np.empty((n * image_size, n * image_size))
        for i in range(n):
            for j in range(n):
                canvas[i*image_size: (i+1)*image_size , j*image_size:(j+1)*image_size] = images[n*i+j].reshape(64,64)
        plt.figure(figsize =(8,8))
        plt.imshow(canvas, cmap ="gray")
        label = "Epoch: {0}".format(order+1)
        plt.xlabel(label)

        if type(order) is str:
            file_name = order
        else:
            file_name = "Mnist_canvas" + str(order)

        plt.savefig(file_name)
        print(os.getcwd())
        print("Image saved in file: ", file_name)
        plt.close()

    def build_net(self):
        self.X = tf.placeholder(tf.float32 , shape = [None, self.length], name ="Input_data")
        self.X_img = tf.reshape(self.X, [-1] + self.data_shape)
        self.z = tf.placeholder(tf.float32, shape = [None, self.z_d], name ="latent_var")

        self.G = self.Generator(self.z, is_training = True, reuse = False)
        self.D_fake_logits = self.Discriminator(self.G, is_training = True, reuse = False)
        self.D_true_logits = self.Discriminator(self.X_img, is_training = True, reuse = True)

        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self. D_fake_logits, labels = tf.ones_like(self.D_fake_logits)))
        self.D_loss_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.D_true_logits , labels = tf.ones_like(self.D_true_logits)))
        self.D_loss_2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.D_fake_logits  , labels = tf.zeros_like(self.D_fake_logits)))
        self.D_loss = self.D_loss_1 + self.D_loss_2

        total_vars = tf.trainable_variables()
        self.d_vars = [var for var in total_vars if  "d_" in var.name]
        self.g_vars = [var for var in total_vars if  "g_" in var.name]


        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.g_optimization = tf.train.AdamOptimizer(learning_rate = self.learning_rate, beta1 = self.beta1).\
                minimize(self.G_loss, var_list = self.g_vars)
            self.d_optimization = tf.train.AdamOptimizer(learning_rate = self.learning_rate, beta1 = self.beta1).\
                minimize(self.D_loss, var_list = self.d_vars)
        print("we successfully make the network")

    def training(self,Data, epoch):
        start_time = time.time()
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())

        for i in range(epoch):
            total_batch = int(len(Data)/self.batch_size)
            d_value = 0
            g_value = 0
            for j in range(total_batch):
                batch_xs = Data[j*self.batch_size:j*self.batch_size + self.batch_size]
                z_sampled1 = np.random.uniform(low  = -1.0, high = 1.0, size = [self.batch_size, self.z_d])
                Op_d, d_= sess.run([self.d_optimization, self.D_loss], feed_dict = {self.X:batch_xs, self.z: z_sampled1})
                z_sampled2 = np.random.uniform(low = -1.0, high = 1.0, size = [self.batch_size, self.z_d])
                Op_g, g_= sess.run([self.g_optimization, self.G_loss], feed_dict = {self.X:batch_xs, self.z: z_sampled2})
                self.images_generated = sess.run(self.G, feed_dict = {self.z:z_sampled2})
                d_value += d_/total_batch
                g_value +=  g_/ total_batch
            self.plot_and_save(i, self.images_generated)
            hour = int((time.time() - start_time)/3600)
            min = int(((time.time() - start_time) - 3600*hour)/60)
            sec = int((time.time()  - start_time) - 3600*hour - 60*min)
            print("Time: ",hour,"h", min,"min",sec ,"sec","   Epoch: ", i, "G_loss: ", g_value, "D_loss: ",d_value)

    def testing(self):
        self.fake_images = self.generator(self.z, is_training = False, reuse = True)
        z_sampled_for_test  = np.random.uniform(low = -1.0, high = 1.0, size = [self.batch_size, self.z_d])
        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        _ = sess.run(self.fake_images, feed_dict  = {self.z: z_sampled_for_test})
        self.plot_and_save("test_image", _)
        print("we have successfully completed the test ")