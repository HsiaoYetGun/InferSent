'''
Created on July 15, 2018
@author : hsiaoyetgun (yqxiao)
Reference : Supervised Learning of Universal Sentence Representations from Natural Language Inference Data (EMNLP 2017)
'''
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, DropoutWrapper
from Utils import print_shape

class InferSent(object):
    def __init__(self, seq_length, n_vocab, embedding_size, hidden_size, attention_size, n_classes, batch_size, learning_rate, optimizer, l2, clip_value):
        # model init
        self._parameter_init(seq_length, n_vocab, embedding_size, hidden_size, attention_size, n_classes, batch_size, learning_rate, optimizer, l2, clip_value)
        self._placeholder_init()

        # model operation
        self.logits = self._logits_op()
        self.loss = self._loss_op()
        self.acc = self._acc_op()
        self.train = self._training_op()

        tf.add_to_collection('train_mini', self.train)

    # init hyper-parameters
    def _parameter_init(self, seq_length, n_vocab, embedding_size, hidden_size, attention_size, n_classes, batch_size, learning_rate, optimizer, l2, clip_value):
        """
        :param seq_length: max sentence length
        :param n_vocab: word nums in vocabulary
        :param embedding_size: embedding vector dims
        :param hidden_size: hidden dims
        :param attention_size: attention dims
        :param n_classes: nums of output label class
        :param batch_size: batch size
        :param learning_rate: learning rate
        :param optimizer: optimizer of training
        :param l2: l2 regularization constant
        :param clip_value: if gradients value bigger than this value, clip it
        """
        self.seq_length = seq_length
        self.n_vocab = n_vocab
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        # Note that attention_size is not used in this model
        self.attention_size = attention_size
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.l2 = l2
        self.clip_value = clip_value

    # placeholder declaration
    def _placeholder_init(self):
        """
        premise_mask: actual length of premise sentence
        hypothesis_mask: actual length of hypothesis sentence
        embed_matrix: with shape (n_vocab, embedding_size)
        dropout_keep_prob: dropout keep probability
        :return:
        """
        self.premise = tf.placeholder(tf.int32, [None, self.seq_length], 'premise')
        self.hypothesis = tf.placeholder(tf.int32, [None, self.seq_length], 'hypothesis')
        self.y = tf.placeholder(tf.float32, [None, self.n_classes], 'y_true')
        self.premise_mask = tf.placeholder(tf.int32, [None], 'premise_actual_length')
        self.hypothesis_mask = tf.placeholder(tf.int32, [None], 'hypothesis_actual_length')
        self.embed_matrix = tf.placeholder(tf.float32, [self.n_vocab, self.embedding_size], 'embed_matrix')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    # build graph
    def _logits_op(self):
        u, v = self._biLSTMMaxEncodingBlock('biLSTM_Max_encoding')
        logits = self._compositionBlock(u, v, self.hidden_size, 'composition')
        return logits

    # feed forward unit
    def _feedForwardBlock(self, inputs, hidden_dims, num_units, scope, isReuse = False, initializer = None):
        """
        :param inputs: tensor with shape (batch_size, 4 * 2 * hidden_size)
        :param scope: scope name
        :return: output: tensor with shape (batch_size, num_units)
        """
        with tf.variable_scope(scope, reuse = isReuse):
            if initializer is None:
                initializer = tf.random_normal_initializer(0.0, 0.1)

            with tf.variable_scope('feed_foward_layer1'):
                inputs = tf.nn.dropout(inputs, self.dropout_keep_prob)
                outputs = tf.layers.dense(inputs, hidden_dims, tf.nn.relu, kernel_initializer = initializer)
            with tf.variable_scope('feed_foward_layer2'):
                outputs = tf.nn.dropout(outputs, self.dropout_keep_prob)
                results = tf.layers.dense(outputs, num_units, tf.nn.tanh, kernel_initializer = initializer)
                return results

    # biLSTM unit
    def _biLSTMBlock(self, inputs, num_units, scope, seq_len = None, isReuse = False):
        with tf.variable_scope(scope, reuse = isReuse):
            lstmCell = LSTMCell(num_units = num_units)
            dropLSTMCell = lambda: DropoutWrapper(lstmCell, output_keep_prob = self.dropout_keep_prob)
            fwLSTMCell, bwLSTMCell = dropLSTMCell(), dropLSTMCell()
            output = tf.nn.bidirectional_dynamic_rnn(cell_fw = fwLSTMCell,
                                                     cell_bw = bwLSTMCell,
                                                     inputs = inputs,
                                                     sequence_length = seq_len,
                                                     dtype = tf.float32)
            return output

    # biLSTM + Max Encoding block
    def _biLSTMMaxEncodingBlock(self, scope):
        """
        :param scope: scope name

        embeded_left, embeded_right: tensor with shape (batch_size, seq_length, embedding_size)
        u_premise, v_hypothesis: concat of biLSTM outputs, tensor with shape (batch_size, timesteps, 2 * hidden_size)

        :return: u: timestep (axis = seq_length) max value of u_premise, tensor with shape (batch_size, 2 * hidden_size)
                 v: timestep (axis = seq_lenght) max value of v_hypothesis, tensor with shape (batch_size, 2 * hidden_size)
        """
        with tf.device('/cpu:0'):
            self.Embedding = tf.get_variable('Embedding', [self.n_vocab, self.embedding_size], tf.float32)
            self.embeded_left = tf.nn.embedding_lookup(self.Embedding, self.premise)
            self.embeded_right = tf.nn.embedding_lookup(self.Embedding, self.hypothesis)
            print_shape('embeded_left', self.embeded_left)
            print_shape('embeded_right', self.embeded_right)

        with tf.variable_scope(scope):
            outputsPremise, finalStatePremise = self._biLSTMBlock(self.embeded_left, self.hidden_size,
                                                                  'biLSTM', self.premise_mask)
            outputsHypothesis, finalStateHypothesis = self._biLSTMBlock(self.embeded_right, self.hidden_size,
                                                              'biLSTM', self.hypothesis_mask,
                                                              isReuse = True)

            u_premise = tf.concat(outputsPremise, axis=2)
            v_hypothesis = tf.concat(outputsHypothesis, axis=2)
            print_shape('u_premise', u_premise)
            print_shape('v_hypothesis', v_hypothesis)

            u = tf.reduce_max(u_premise, axis=1)
            v = tf.reduce_max(v_hypothesis, axis=1)
            print_shape('u', u)
            print_shape('v', v)
            return u, v

    # composition block
    def _compositionBlock(self, u, v, hiddenSize, scope):
        """
        :param u: timestep (axis = seq_length) max value of u_premise, tensor with shape (batch_size, 2 * hidden_size)
        :param v: timestep (axis = seq_lenght) max value of v_hypothesis, tensor with shape (batch_size, 2 * hidden_size)
        :param hiddenSize: biLSTM cell's hidden states size
        :param scope: scope name

        diff: absolute difference of u and v, tensor with shape (batch_size, 2 * hidden_size)
        mul: hadamard product of u and v, tensor with shape (batch_size, 2 * hidden_size)
        features: concat of [u, v, diff, mul], tensor with shape (batch_size, 4 * 2 * hidden_size)

        :return: y_hat: output of feed forward layer, tensor with shape (batch_size, n_classes)
        """
        with tf.variable_scope(scope):
            diff = tf.abs(tf.subtract(u, v))
            mul = tf.multiply(u, v)
            print_shape('diff', diff)
            print_shape('mul', mul)

            features = tf.concat([u, v, diff, mul], axis=1)
            print_shape('features', features)

            y_hat = self._feedForwardBlock(features, self.hidden_size, self.n_classes, 'feed_forward')
            print_shape('y_hat', y_hat)
            return y_hat

    # calculate classification loss
    def _loss_op(self, l2_lambda=0.0001):
        with tf.name_scope('cost'):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.logits)
            loss = tf.reduce_mean(losses, name='loss_val')
            weights = [v for v in tf.trainable_variables() if ('w' in v.name) or ('kernel' in v.name)]
            l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights]) * l2_lambda
            loss += l2_loss
        return loss

    # calculate classification accuracy
    def _acc_op(self):
        with tf.name_scope('acc'):
            label_pred = tf.argmax(self.logits, 1, name='label_pred')
            label_true = tf.argmax(self.y, 1, name='label_true')
            correct_pred = tf.equal(tf.cast(label_pred, tf.int32), tf.cast(label_true, tf.int32))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='Accuracy')
        return accuracy

    # define optimizer
    def _training_op(self):
        with tf.name_scope('training'):
            if self.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
            elif self.optimizer == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            elif self.optimizer == 'momentum':
                optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9)
            elif self.optimizer == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            elif self.optimizer == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
            elif self.optimizer == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(self.learning_rate)
            else:
                ValueError('Unknown optimizer : {0}'.format(self.optimizer))
        gradients, v = zip(*optimizer.compute_gradients(self.loss))
        if self.clip_value is not None:
            gradients, _ = tf.clip_by_global_norm(gradients, self.clip_value)
        train_op = optimizer.apply_gradients(zip(gradients, v))
        return train_op

    # learning rate decay
    def _learning_rate_decay_op(self):
        return self.learning_rate.assign(self.learning_rate * 0.2)