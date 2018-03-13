import tensorflow as tf
import data
import model

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer("z_d", 100, "Dimension of z")
flags.DEFINE_float("learning_rate", 0.0002, "learning_rate")
flags.DEFINE_integer("batch_size", 64, "batch_size")
flags.DEFINE_integer("n_epoch", 100, "number of epoch")

train_data,train_labels,validation_data,validation_labels,test_data,test_labels = data.prepare_MNIST_Data()
train_data = data.preprocessing_data(train_data)
sess = tf.Session()
network = model.Gan(sess,[64,64,1],FLAGS.z_d, FLAGS.learning_rate, FLAGS.batch_size)
sess.run(tf.initialize_all_variables())
network.training(train_data,FLAGS.n_epoch)
network.testing()
