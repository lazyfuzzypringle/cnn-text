import tensorflow as tf
import numpy as np


def cosine(x1, x2):
    norm_x1 = tf.nn.l2_normalize(x1,0)
    norm_x2 = tf.nn.l2_normalize(x2,0)
    # return tf.reduce_sum(tf.multiply(norm_x1, norm_x2),-1)
    return tf.losses.cosine_distance(norm_x1, norm_x2, axis=-1)


def eucidean(x1, x2):
    return tf.sqrt(tf.reduce_sum(tf.square(tf.sub(x1,x2)), reduction_indices=1))


SCORE_FUNC = {'cosine': cosine, 'eucidean': eucidean}


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size, embedding_weights,ydim, filter_sizes, num_filters,
            l2_reg_lambda=0.0, score_function='cosine', margin=1):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # self.W = tf.Variable(
            #     tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
            #     name="W")
            self.W = tf.get_variable(name="embedding_weights", shape=[vocab_size, embedding_size],
                                     initializer=tf.constant_initializer(embedding_weights),
                                     trainable=True)

            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        with tf.name_scope("loss_anchor"):
            # self.xp = tf.Variable('xp',shape=[ydim], initializer=tf.random_normal_initializer())
            self.xn = tf.get_variable('xn',shape=[1, ydim], initializer=tf.random_normal_initializer())
            self.xp = tf.get_variable('xp',shape=[1, ydim], initializer=tf.random_normal_initializer())

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, ydim],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[ydim]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.logit = tf.nn.xw_plus_b(self.h_drop, W, b) #batch, ydim
            batch_size = tf.shape(self.logit)[0]
            xp = tf.tile(self.xp, [batch_size,1])
            xn = tf.tile(self.xn, [batch_size,1])
            pos_scores = SCORE_FUNC[score_function](self.logit, xp)
            neg_scores = SCORE_FUNC[score_function](self.logit, xn)
            self.predictions = tf.cast(tf.less(pos_scores,neg_scores), dtype=tf.int64)


        # FIXME loss
        with tf.name_scope("loss"):
            # for 3 instances with labels:
            #  [0    1   0] neg, pos, neg
            # labels:
            #  [-1   1  -1]
            # losses are:
            # -pos+neg
            # +pos-neg
            # +neg-pos
            # pos_distance: 0.2 0.3 0.4 0.5
            # neg_distance: 0.3 0.4 0.5 0.7
            # pos*label+neg*(-label)
            labels = 2 * self.input_y-1
            neg_scores = tf.multiply(neg_scores, labels)
            pos_scores = tf.multiply(pos_scores, labels)
            losses = tf.maximum(0., margin + pos_scores - neg_scores)

            # losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")