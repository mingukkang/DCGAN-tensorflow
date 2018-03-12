## Deep Convolutional Generative Adversarial Networks(DCGAN)

**I Write a tensorflow code for DCGAN that generates MNIST Data.**

**This code has following features**
1. latent Vector is 100 dimensions.
2. We do not applying batch_normalization to the generator output layer, But we apply batch_normalization to the discriminator input layer.

(※ In the paper, the author claims not applying batchnorm to the generator output layer and discriminator input layer.)
3. We resized MNIST_Data to 64x64x1 and normalized it.


## Enviroment
**- OS: window 10(64bit)**

**- Python 3.5**

**- Tensorflow-gpu version:  1.3.0rc2**

## Architecture Guidelines for Stable Deep Convolutional GANs
![사진1](https://github.com/MINGUKKANG/DCGAN_tensorflow/blob/master/images/guideline.JPG)

## Schematic of DCGAN
![사진2](https://github.com/MINGUKKANG/DCGAN_tensorflow/blob/master/images/schemetic.JPG)

## Code

**1. Generator**
```python
def Generator(self, z, is_training, reuse):
    depths = [1024, 512, 256, 128] + [self.data_shape[2]]
    with tf.variable_scope("Generator", reuse = reuse):
        with tf.variable_scope("g_fc1", reuse = reuse):
            output = tf.layers.dense(z, depths[0]*4*4, trainable = is_training)
            output = tf.reshape(output, [self.batch_size, 4, 4, depths[0]])
            output = tf.nn.relu(tf.layers.batch_normalization(output, training = is_training))

        with tf.variable_scope("g_dc1", reuse = reuse):
            output = tf.layers.conv2d_transpose(output, depths[1], [5,5], strides =(2,2), padding ="SAME",
                                                trainable = is_training)
            output = tf.nn.relu(tf.layers.batch_normalization(output, training = is_training))

        with tf.variable_scope("g_dc2", reuse = reuse):
            output = tf.layers.conv2d_transpose(output, depths[2], [5,5], strides = (2,2), padding ="SAME", 
                                                trainable = is_training)
            output = tf.nn.relu(tf.layers.batch_normalization(output, training = is_training))

        with tf.variable_scope("g_dc3", reuse = reuse):
            output = tf.layers.conv2d_transpose(output,depths[3], [5,5], strides = (2,2), padding ="SAME",
                                                trainable = is_training)
            output = tf.nn.relu(tf.layers.batch_normalization(output, training = is_training))

        with tf.variable_scope("g_dc4", reuse = reuse):
            output = tf.layers.conv2d_transpose(output,depths[4], [5,5], strides = (2,2), padding = "SAME", 
                                                trainable = is_training)
            self.g_logits = tf.nn.tanh(output)

    return self.g_logits
```
**2. Discriminator**
```python
def Discriminator(self,X, is_training, reuse):
    depths = [self.data_shape[2]] + [64, 128, 256, 512]
    with tf.variable_scope("Discriminator", reuse = reuse):
        with tf.variable_scope("d_cv1", reuse = reuse):
            output = tf.layers.conv2d(X, depths[1], [5,5], strides = (2,2), padding ="SAME", trainable = is_training)
            output = tf.nn.leaky_relu(tf.layers.batch_normalization(output, training = is_training))

        with tf.variable_scope("d_cv2", reuse = reuse):
            output = tf.layers.conv2d(output, depths[2], [5,5], strides = (2,2), padding ="SAME", 
                                      trainable = is_training)
            output = tf.nn.leaky_relu(tf.layers.batch_normalization(output, training = is_training))

        with tf.variable_scope("d_cv3", reuse = reuse):
            output = tf.layers.conv2d(output, depths[3], [5,5], strides = (2,2), padding = "SAME", 
                                      trainable = is_training)
            output = tf.nn.leaky_relu(tf.layers.batch_normalization(output, training = is_training))

        with tf.variable_scope("d_cv4", reuse = reuse):
            output = tf.layers.conv2d(output, depths[4], [5,5], strides = (2,2), padding ="SAME", 
                                      trainable = is_training)
            output = tf.nn.leaky_relu(tf.layers.batch_normalization(output, training = is_training))

        with tf.variable_scope("d_fc1", reuse = reuse):
            output = tf.layers.flatten(output)
            self.d_logits = tf.layers.dense(output,1, trainable = is_training)

    return self.d_logits
```

**3. Deep convolutional Generative Adversarial Networks**
```python
def build_net(self):
    self.X = tf.placeholder(tf.float32 , shape = [None, self.length], name ="Input_data")
    self.X_img = tf.reshape(self.X, [-1] + self.data_shape)
    self.z = tf.placeholder(tf.float32, shape = [None, self.z_d], name ="latent_var")

    self.G = self.Generator(self.z, is_training = True, reuse = False)
    self.D_fake_logits = self.Discriminator(self.G, is_training = True, reuse = False)
    self.D_true_logits  = self.Discriminator(self.X_img, is_training = True, reuse = True)

    self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self. D_fake_logits, 
                                                                         labels = tf.ones_like(self.D_fake_logits)))
    self.D_loss_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.D_true_logits,
                                                                           labels = tf.ones_like(self.D_true_logits)))
    self.D_loss_2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = self.D_fake_logits,
                                                                           labels = tf.zeros_like(self.D_fake_logits)))
    self.D_loss = self.D_loss_1 + self.D_loss_2

    total_vars = tf.trainable_variables()
    self.d_vars = [var for var in total_vars if  "d_" in var.name]
    self.g_vars = [var for var in total_vars if  "g_" in var.name]


    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        self.g_optimization = tf.train.AdamOptimizer(learning_rate = self.learning_rate, beta1 = self.beta1).\
            minimize(self.G_loss, var_list = self.g_vars)
        self.d_optimization = tf.train.AdamOptimizer(learning_rate = self.learning_rate, beta1 = self.beta1).\
            minimize(self.D_loss, var_list = self.d_vars)
```

## Result
**1. Comparing the generated images with the original images(n_z = 10)**

![사진5](https://github.com/MINGUKKANG/VAE_tensorflow/blob/master/image/Result1.PNG)

**2. Result, When n_z =2**
<table align='center'>
<tr align='center'>
<td> Distribution of MNIST </td>
<td> Manifold of MNIST </td>
</tr>
<tr>
<td><img src = 'image/result2.JPG' height = '400px'>
<td><img src = 'image/result3.JPG' height = '400px'>
</tr>
</table>

## Reference Papers
**1. https://arxiv.org/abs/1312.6114**

**2. https://arxiv.org/abs/1606.05908**

## References

**1.https://github.com/hwalsuklee/tensorflow-mnist-VAE**

**2.https://github.com/shaohua0116/VAE-Tensorflow**

**3.http://cs231n.stanford.edu/**

**4.https://www.facebook.com/groups/TensorFlowKR/permalink/496009234073473/?hc_location=ufi**

**-- Above Reference is ppt which is distributed by Hwal-Suk Lee from facebook page tensorflow korea**

**5.http://jaejunyoo.blogspot.com/2017/04/auto-encoding-variational-bayes-vae-1.html**
