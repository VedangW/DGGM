#!usr/bin/python

""" 
Class definition of class AAE, which defines the adversarial
autoencoder. Also contains the training and loading methods.
"""

import config
import numpy as np
import tensorflow as tf

from aae_func import batch_gen, buffered_gen, load_test, he_initializer, linear_layer, bn_layer, sample_prior, train, test

# Initialise global variables according to hyperparameters in config

disc_power = config.disc_power
gpu_config_protocol = config.gpu_config

batch_size_for_model = config.batch_size_for_model
num_of_test_instances = config.num_of_test_instances

input_layer_size = config.input_layer_size
latent_layer_size = config.latent_layer_size
output_layer_size = config.output_layer_size
middle_layer_sizes = config.middle_layer_sizes

discriminator_sizes = config.discriminator_sizes

# Class definition for AAE

class AAE(object):
    """ Class which defines the Adversarial Autoencoder. """

    def __init__(self,
                 learning_rate=0.001,
                 discriminative_power=disc_power,
                 gpu_config_proto=gpu_config_protocol,
                 batch_size=batch_size_for_model, 
                 input_space=input_layer_size,
                 latent_space=latent_layer_size,
                 middle_layers=middle_layer_sizes,
                 disc_layers=discriminator_sizes,
                 activation_fn=tf.nn.tanh,
                 initializer=he_initializer):

        self.batch_size = batch_size
        self.input_space = input_space
        self.latent_space = latent_space
        if middle_layers is None:
            self.middle_layers = [256, 256]
        else:
            self.middle_layers = middle_layers
        self.activation_fn = activation_fn
        self.learning_rate = learning_rate

        self.initializer = initializer

        tf.reset_default_graph()
        
        self.input_x = tf.placeholder(tf.float32, [None, input_space])
        self.z_tensor = tf.placeholder(tf.float32, [None, latent_space])

        # Encoder net 
        # Original sizes: 167->256->256->20
        with tf.variable_scope("encoder"):
            self.encoder_layers = self.encoder()
            self.encoded = self.encoder_layers[-1]
        
        # Decoder net
        # Original sizes: 20->256->256->167
        with tf.variable_scope("decoder"):
            self.decoder_layers = self.decoder(self.encoded)
            self.decoded = self.decoder_layers[-1]
            tf.get_variable_scope().reuse_variables()
            self.generator_layers = self.decoder(self.z_tensor)
            self.generated = tf.nn.sigmoid(self.generator_layers[-1])

        # Discriminator net
        # Original sizes: 20->64->64->8->1
        sizes = disc_layers
        with tf.variable_scope("discriminator"):
            self.disc_layers_neg = self.discriminator(self.encoded, sizes)
            self.disc_neg = self.disc_layers_neg[-1]
            tf.get_variable_scope().reuse_variables()
            self.disc_layers_pos = self.discriminator(self.z_tensor, sizes)
            self.disc_pos = self.disc_layers_pos[-1]

        self.pos_loss = tf.nn.relu(self.disc_pos) - self.disc_pos + tf.log(1.0 + tf.exp(-tf.abs(self.disc_pos)))
        self.neg_loss = tf.nn.relu(self.disc_neg) + tf.log(1.0 + tf.exp(-tf.abs(self.disc_neg)))
        self.disc_loss = tf.reduce_mean(tf.add(self.pos_loss, self.neg_loss))
        
        tf.summary.scalar("discriminator_loss", self.disc_loss)
            
        self.enc_loss = tf.reduce_mean(tf.nn.relu(self.disc_neg) - self.disc_neg + tf.log(1.0 + tf.exp(-tf.abs(self.disc_neg))))
        batch_logloss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.decoded, labels=self.input_x), 1)
        self.dec_loss = tf.reduce_mean(batch_logloss)
        
        tf.summary.scalar("encoder_loss", self.enc_loss)
        tf.summary.scalar("decoder_loss", self.dec_loss)
        
        disc_ws = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
        enc_ws = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder')
        ae_ws = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder') + \
                tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder')
            
        self.train_discriminator = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.disc_loss, var_list=disc_ws)
        self.train_encoder = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.enc_loss, var_list=enc_ws)
        self.train_autoencoder = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.dec_loss, var_list=ae_ws)

        if gpu_config_proto[0] != 'NONE':
            if gpu_config_proto[0] == 'GPU_MEM_FRACTION':
                gpu_config = tf.ConfigProto()
                gpu_config.gpu_options.per_process_gpu_memory_fraction = gpu_config_proto[1]
            elif gpu_config_proto[0] == 'ALLOW_GROWTH':
                gpu_config = tf.ConfigProto()
                gpu_config.gpu_options.allow_growth = True
            
            self.sess = tf.Session(config=gpu_config)
            self.init_net()
        else:
            self.sess = tf.Session()
            self.init_net
        
    def encoder(self):
        sizes = self.middle_layers + [self.latent_space]
        with tf.variable_scope("layer-0"):
            encoder_layers = [linear_layer(self.input_x, self.input_space, sizes[0])]
        for i in range(len(sizes) - 1):
            with tf.variable_scope("layer-%i" % (i+1)):
                activated = self.activation_fn(encoder_layers[-1])
                normed = bn_layer(activated, sizes[i])
                next_layer = linear_layer(activated, sizes[i], sizes[i+1])
            encoder_layers.append(next_layer)
            
        return encoder_layers

    def decoder(self, tensor):
        sizes = self.middle_layers[::-1] + [self.input_space]
        with tf.variable_scope("layer-0"):
            decoder_layers = [linear_layer(tensor, self.latent_space, sizes[0])]
        for i in range(len(sizes) - 1):
            with tf.variable_scope("layer-%i" % (i+1)):
                activated = self.activation_fn(decoder_layers[-1])
                normed = bn_layer(activated, sizes[i])
                next_layer = linear_layer(activated, sizes[i], sizes[i+1])
            decoder_layers.append(next_layer)
        
        return decoder_layers
    
    def discriminator(self, tensor, sizes):
        with tf.variable_scope("layer-0"):
            disc_layers = [linear_layer(tensor, self.latent_space, sizes[0])]
        for i in range(len(sizes) - 1):
            with tf.variable_scope("layer-%i" % (i+1)):
                activated = tf.nn.tanh(disc_layers[-1])
#                 normed = bn_layer(activated, sizes[i])
                next_layer = linear_layer(activated, sizes[i], sizes[i+1])
            disc_layers.append(next_layer)

        return disc_layers
    
    def init_net(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)        
    
    def train(self, log, summary_file, epochs):
        sess = self.sess
        saver = tf.train.Saver()
        hist = []
        test_data = load_test()
        
        merged_summary = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(summary_file)
        train_writer.add_graph(sess.graph)
        
        for e in range(epochs):
            print ("Epoch: ", e+1)
            print (log, "epoch #%d" % (e+1))
            log.flush()
            train_gen = buffered_gen(train, batch_n=self.batch_size)
            for i, batch_x in enumerate(train_gen):
                if i % 3 == 0:
                    batch_z = sample_prior(scale=1.0, size=(len(batch_x), self.latent_space))
                    sess.run(self.train_discriminator, feed_dict={self.input_x: batch_x, self.z_tensor: batch_z})
                elif i % 3 == 1:
                    sess.run(self.train_encoder, feed_dict={self.input_x: batch_x})
                else:
                    sess.run(self.train_autoencoder, feed_dict={self.input_x: batch_x})
                if i % 5 == 0:
                    batch_z = sample_prior(scale=1.0, size=(len(test_data), self.latent_space))
                    s = sess.run(merged_summary, feed_dict={self.input_x: test_data, self.z_tensor: batch_z})
                    train_writer.add_summary(s, i)
                if i % 10000 == 0:
                    batch_z = sample_prior(scale=1.0, size=(len(test_data), self.latent_space))
                    
                    losses = sess.run([self.disc_loss, self.enc_loss, self.dec_loss],
                                      feed_dict={self.input_x: test_data, self.z_tensor: batch_z})
                    discriminator_loss, encoder_loss, decoder_loss = losses
                    print (log, "disc: %f, encoder : %f, decoder : %f" % (discriminator_loss/2., encoder_loss, decoder_loss))
                    
                    log.flush()
            else:
                if e == epochs-1:     
                    saver.save(sess, './fpt.aae.%de.model.ckpt' % e)
                batch_z = sample_prior(scale=1.0, size=(len(test_data), self.latent_space))
                losses = sess.run([self.disc_loss, self.enc_loss, self.dec_loss],
                                  feed_dict={self.input_x: test_data, self.z_tensor: batch_z})


                discriminator_loss, encoder_loss, decoder_loss = losses
                print (log, "disc: %f, encoder : %f, decoder : %f" % (discriminator_loss/2., encoder_loss, decoder_loss))
                    
                log.flush()
                hist.append(decoder_loss)
        return hist
    
    def load(self, model):
        saver = tf.train.Saver()
        saver.restore(self.sess, model)