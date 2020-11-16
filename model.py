# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from collections import OrderedDict

class Sequence_Embedding(tf.keras.layers.Layer):
    def __init__(self, clicked_item_dim, pos_item_dim, unclick_item_dim, feedback_item_dim, item_dim,
                 initializers=tf.keras.initializers.GlorotNormal(), **kwargs):
        super().__init__(**kwargs)
        self.clicked_item_dim = clicked_item_dim
        self.pos_item_dim = pos_item_dim
        self.unclick_item_dim = unclick_item_dim
        self.feedback_item_dim = feedback_item_dim
        self.item_dim = item_dim
        self.pos_w_clicked = self.add_weight(name="pos_w_clicked", shape=(self.clicked_item_dim + self.pos_item_dim, self.item_dim), 
                                             initializer=initializers,
                                             dtype=tf.float32)
        self.pos_w_unclick = self.add_weight(name="pos_w_unclick", shape=(self.unclick_item_dim + self.pos_item_dim, self.item_dim), 
                                             initializer=initializers,
                                             dtype=tf.float32)
        self.pos_w_feedback = self.add_weight(name="pos_w_feedback", shape=(self.feedback_item_dim + self.pos_item_dim, self.item_dim), 
                                             initializer=initializers,
                                             dtype=tf.float32)

    def call(self, inputs, **kwargs):
        clicked_z = tf.matmul(inputs[0], self.pos_w_clicked)
        unclick_z = tf.matmul(inputs[1], self.pos_w_unclick)
        feedback_z = tf.matmul(inputs[2], self.pos_w_feedback)
        return [clicked_z, unclick_z, feedback_z]

    def compute_output_shape(self, input_shape):
        return [(None, self.item_dim), (None, self.item_dim), (None, self.item_dim)]

class Embedding_Lookup(tf.keras.layers.Layer):
    def __init__(self, feature_size, embed_dim, initializers=tf.keras.initializers.GlorotNormal(), **kwargs):
        super().__init__(**kwargs)
        self.feature_size = feature_size
        self.embed_dim = embed_dim
        self.initializers = initializers
        
    def build(self, input_shape):
        super().build(input_shape)
        self.embedding_w = self.add_weight(name="embedding_w", shape=(self.feature_size, self.embed_dim), 
                                           initializer=self.initializers,
                                          )
        
    def call(self, inputs, **kwargs):
        embedding = tf.nn.embedding_lookup_sparse(self.embedding_w, inputs, sp_weights=None, combiner='mean')
        return embedding

    def compute_output_shape(self, input_shape):
        return (None, self.embed_dim)

class Transformer(tf.keras.layers.Layer):
    def __init__(self, hist_size, hist_embedding_dim, prefix="",initializers=tf.keras.initializers.GlorotNormal(), **kwargs):
        super().__init__(**kwargs)
        self.headnum = 4
        self.hist_size = hist_size
        self.hist_embedding_dim = hist_embedding_dim
        self.initializers = initializers
        self.prefix = prefix
        self.attQ_w = [self.add_weight(name=prefix + "attQ_w" + str(i), shape=(self.hist_embedding_dim, int(self.hist_embedding_dim / self.headnum)), 
                                       initializer=self.initializers, dtype=tf.float32) for i in range(self.headnum)]
        self.attK_w = [self.add_weight(name=prefix + "attK_w" + str(i), shape=(self.hist_embedding_dim, int(self.hist_embedding_dim / self.headnum)), 
                                       initializer=self.initializers, dtype=tf.float32) for i in range(self.headnum)]
        self.attV_w = [self.add_weight(name=prefix + "attV_w" + str(i), shape=(self.hist_embedding_dim, int(self.hist_embedding_dim / self.headnum)), 
                                      initializer=self.initializers, dtype=tf.float32) for i in range(self.headnum)]

        self.FFN_w0 = self.add_weight(name=prefix + 'FFN_w0', shape=(self.hist_embedding_dim, self.hist_embedding_dim * 4), 
                                      initializer=self.initializers, dtype=tf.float32)
        self.FFN_b0 = self.add_weight(name=prefix + 'FFN_b0', shape=(self.hist_embedding_dim * 4,), 
                                      initializer=self.initializers, dtype=tf.float32)

        self.FFN_w1 = self.add_weight(name=prefix + 'FFN_w1', shape=(self.hist_embedding_dim * 4, self.hist_embedding_dim), 
                                      initializer=self.initializers, dtype=tf.float32)
        self.FFN_b1 = self.add_weight(name=prefix + 'FFN_b1', shape=(self.hist_embedding_dim,), 
                                      initializer=self.initializers, dtype=tf.float32)
        
    def call(self, inputs, **kwargs):
        candidate_embedding, hist_embeddings, hisLens = inputs
        hist_size = self.hist_size + 1
        hist_z = [candidate_embedding]
        for i in range(0, self.hist_size):
            hist_z.append(hist_embeddings[i])
        hist_z_all = tf.stack(hist_z, axis=1) #(batch, hist_size, hist_embedding_dim)
        
        mutil_head_att = []

        #attention
        for i in range(0, self.headnum):
            
            attQ = tf.tensordot(hist_z_all, self.attQ_w[i], axes=1) #(batch, hist_size, hist_embedding_dim/headnum)
            attK = tf.tensordot(hist_z_all, self.attK_w[i], axes=1) #(batch, hist_size, hist_embedding_dim/headnum)
            attV = tf.tensordot(hist_z_all, self.attV_w[i], axes=1) #(batch, hist_size, hist_embedding_dim/headnum)
            
            attQK = tf.matmul(attQ, attK, transpose_b=True) #(batch, hist_size, hist_size)

            #scale
            attQK_scale = attQK / (self.hist_embedding_dim ** 0.5)
            padding = tf.ones_like(attQK_scale) * (-2 ** 32 + 1) #(batch, hist_size, hist_size)

            #mask
            key_masks = tf.sequence_mask(hisLens + 1, hist_size)  # (batch, 1, hist_size)
            key_masks_new = tf.reshape(key_masks, [-1, 1, hist_size])
            key_masks_tile = tf.tile(key_masks_new, [1, hist_size, 1]) #(batch, hist_size, hist_size)
            key_masks_cast = tf.cast(key_masks_tile, dtype=tf.float32)
            outputs_QK = tf.where(key_masks_tile, attQK_scale, padding) #(batch, hist_size, hist_size)

            #norm
            outputs_QK_norm = tf.nn.softmax(outputs_QK) #(batch, hist_size, hist_size)

            #query mask
            outputs_QK_q = tf.multiply(outputs_QK_norm, key_masks_cast) #(batch, hist_size, hist_size)
            # weighted sum
            outputs_QKV_head = tf.matmul(outputs_QK_q, attV) #(batch, hist_size, hist_embedding_dim/headnum)
            mutil_head_att.append(outputs_QKV_head)

        outputs_QKV = tf.concat(mutil_head_att, axis=2) # (batch, hist_size, hist_embedding_dim)
        #FFN
        TH0 = tf.tensordot(outputs_QKV, self.FFN_w0, axes=1) + self.FFN_b0 #(batch, hist_size, hist_embedding_dim * 4)
        TZ0 = tf.nn.relu(TH0)
        TH1 = tf.tensordot(TZ0, self.FFN_w1, axes=1) + self.FFN_b1 # (batch, hist_size, hist_embedding_dim)

        return tf.reduce_sum(TH1, axis=1) #(batch, hist_embedding_dim)
    
    def compute_output_shape(self, input_shape):
        return (None, self.hist_embedding_dim)

class Attention(tf.keras.layers.Layer):
    def __init__(self, hist_size, hist_embedding_dim, prefix="", initializers=tf.keras.initializers.GlorotNormal(), **kwargs):
        super().__init__(**kwargs)
        self.hist_size = hist_size
        self.hist_embedding_dim = hist_embedding_dim
        self.initializers = initializers
        self.attention_hidden_ = 32
        self.attW1 = self.add_weight(name=prefix + "attention_hidden_w1", shape=(self.hist_embedding_dim * 4, self.attention_hidden_), 
                                     initializer=self.initializers, dtype=tf.float32)
        self.attB1 = self.add_weight(name=prefix + "attention_hidden_b1", shape=(self.attention_hidden_,), 
                                     initializer=tf.keras.initializers.Zeros(), dtype=tf.float32)

        self.attW2 = self.add_weight(name=prefix + "attention_hidden_w2", shape=(self.attention_hidden_, 1), 
                                     initializer=self.initializers, dtype=tf.float32)
        self.attB2 = self.add_weight(name=prefix + "attention_hidden_b2", shape=(1,), 
                                     initializer=tf.keras.initializers.Zeros(), dtype=tf.float32)

    def call(self, inputs, **kwargs):
        candidate_embedding, hist_embeddings, hisLens = inputs
        
        hist_embedding_list = []
        for i in range(0, self.hist_size):
            # batch, hist_embedding_dim * 4
            z1 = tf.concat([candidate_embedding, hist_embeddings[i], candidate_embedding * hist_embeddings[i], candidate_embedding - hist_embeddings[i]], axis=1)
            hist_embedding_list.append(z1)
        hist_z_all = tf.stack(hist_embeddings, axis=1) #(batch, hist_size, hist_embedding_dim)

        z2 = tf.concat(hist_embedding_list, axis=1)  #(batch, hist_size * hist_embedding_dim * 4)

        z3 = tf.reshape(z2, [-1, self.hist_size, 4 * self.hist_embedding_dim])

        z4 = tf.matmul(z3, self.attW1) + self.attB1
        z5 = tf.nn.relu(z4)
        z6 = tf.matmul(z5, self.attW2) + self.attB2
        att_w_all = tf.squeeze(z6, axis=2)

        # mask
        hist_masks = tf.sequence_mask(hisLens, self.hist_size) #(batch, hist_size)
        padding = tf.ones_like(att_w_all) * (-2**32 + 1)
        att_w_all_rep = tf.where(hist_masks, att_w_all, padding)

        # scale
        att_w_all_scale = att_w_all_rep / (self.hist_embedding_dim ** 0.5)

        # norm
        att_w_all_norm = tf.nn.softmax(att_w_all_scale)

        att_w_all_mul = tf.reshape(att_w_all_norm, [-1, 1, self.hist_size])

        weighted_hist_all = tf.matmul(att_w_all_mul, hist_z_all) #(batch, 1, hist_embedding_dim)

        return tf.squeeze(weighted_hist_all, axis=1)

    def compute_output_shape(self, input_shape):
        return (None, self.hist_embedding_dim)

class DNN(tf.keras.layers.Layer):
    """The Multi Layer Percetron

      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.

      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.

      Arguments
        - **hidden_units**:list of positive integer, the layer number and units in each layer.

        - **activation**: Activation function to use.

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.

        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.

        - **use_bn**: bool. Whether use BatchNormalization before activation or not.

        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False, seed=1024, **kwargs):
        super(DNN, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        self.activation_dict = {"relu": tf.keras.activations.relu,
                                "sigmoid": tf.keras.activations.sigmoid,
                                "tanh": tf.keras.activations.tanh}

    def build(self, input_shape):
        super(DNN, self).build(input_shape)  # Be sure to call this somewhere!
        # if len(self.hidden_units) == 0:
        #     raise ValueError("hidden_units is empty")
        input_size = input_shape[-1]
        hidden_units = [int(input_size)] + list(self.hidden_units)
        self.kernels = [self.add_weight(name='kernel' + str(i),
                                        shape=(
                                            hidden_units[i], hidden_units[i + 1]),
                                            initializer=tf.keras.initializers.GlorotNormal(
                                                seed=self.seed),
                                        regularizer=tf.keras.regularizers.l2(self.l2_reg),
                                        trainable=True) for i in range(len(self.hidden_units))]
        self.bias = [self.add_weight(name='bias' + str(i),
                                     shape=(self.hidden_units[i],),
                                     initializer=tf.keras.initializers.Zeros(),
                                     trainable=True) for i in range(len(self.hidden_units))]
        if self.use_bn:
            self.bn_layers = [tf.keras.layers.BatchNormalization() for _ in range(len(self.hidden_units))]

        self.dropout_layers = [tf.keras.layers.Dropout(self.dropout_rate, seed=self.seed + i) for i in
                               range(len(self.hidden_units))]

        self.activation_layers = [self.activation_dict[self.activation] for _ in range(len(self.hidden_units))]


    def call(self, inputs, training=None, **kwargs):

        deep_input = inputs

        for i in range(len(self.hidden_units)):
            fc = tf.nn.bias_add(tf.tensordot(
                deep_input, self.kernels[i], axes=(-1, 0)), self.bias[i])
            # fc = Dense(self.hidden_size[i], activation=None, \
            #           kernel_initializer=glorot_normal(seed=self.seed), \
            #           kernel_regularizer=l2(self.l2_reg))(deep_input)
            if self.use_bn:
                fc = self.bn_layers[i](fc, training=training)

            fc = self.activation_layers[i](fc)

            fc = self.dropout_layers[i](fc, training=training)
            deep_input = fc

        return deep_input

    def compute_output_shape(self, input_shape):
        if len(self.hidden_units) > 0:
            shape = input_shape[:-1] + (self.hidden_units[-1],)
        else:
            shape = input_shape

        return tuple(shape)

    def get_config(self, ):
        config = {'activation': self.activation, 'hidden_units': self.hidden_units,
                  'l2_reg': self.l2_reg, 'use_bn': self.use_bn, 'dropout_rate': self.dropout_rate, 'seed': self.seed}
        base_config = super(DNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DFN():
    def __init__(self, main_group_ids, candidate_group_ids, clicked_group_ids, unclick_group_ids, feedback_group_ids, 
                 pos_group_ids, batch_size=256, embed_dim=16, feature_size=1048573, hist_size=30):
        self.main_group_ids = main_group_ids
        self.candidate_group_ids = candidate_group_ids
        self.clicked_group_ids = clicked_group_ids
        self.unclick_group_ids = unclick_group_ids
        self.feedback_group_ids = feedback_group_ids
        self.pos_group_ids = pos_group_ids
        self.batch_size = batch_size
        self.embed_dim = embed_dim
        self.feature_size = feature_size
        self.hist_size = hist_size
        self.group_feature = OrderedDict()
        self.clicked_item_dim = len(clicked_group_ids) * embed_dim
        self.unclick_item_dim = len(unclick_group_ids) * embed_dim
        self.feedback_item_dim = len(feedback_group_ids) * embed_dim
        self.item_dim = self.clicked_item_dim
        self.pos_item_dim = len(pos_group_ids) * embed_dim
        self._results = None
        # build input
        for group_id in main_group_ids:
          self.group_feature["main_" + str(group_id)] = tf.keras.layers.Input(shape=(self.feature_size, ), dtype=tf.int32, sparse=True, name=("main_" + str(group_id)))
        for group_id in candidate_group_ids:
          self.group_feature["candidate_" + str(group_id)] = tf.keras.layers.Input(shape=(self.feature_size, ), dtype=tf.int32, sparse=True, name=("candidate_" + str(group_id)))
 
        for i in range(0, hist_size):
            for group_id in clicked_group_ids:
                self.group_feature["clicked" + "_" + str(i) + "_" + str(group_id)] = tf.keras.layers.Input(shape=(self.feature_size, ), dtype=tf.int32, sparse=True, name=("clicked" + "_" + str(i) + "_" + str(group_id)))
            for group_id in unclick_group_ids:
                self.group_feature["unclick" + "_" + str(i) + "_" + str(group_id)] = tf.keras.layers.Input(shape=(self.feature_size, ), dtype=tf.int32, sparse=True, name=("unclick" + "_" + str(i) + "_" + str(group_id)))
            for group_id in feedback_group_ids:
                self.group_feature["feedback" + "_" + str(i) + "_" + str(group_id)] = tf.keras.layers.Input(shape=(self.feature_size, ), dtype=tf.int32, sparse=True, name=("feedback" + "_" + str(i)+"_"+str(group_id)))  
            for group_id in pos_group_ids:
                self.group_feature["clicked" + "_" + "position" + "_" + str(i) + "_" + str(group_id)] = tf.keras.layers.Input(shape=(self.feature_size, ), dtype=tf.int32, sparse=True, name=("clicked" + "_" + "position" + "_"+str(i) + "_"+str(group_id)))
                self.group_feature["unclick" + "_" + "position" + "_" + str(i) + "_" + str(group_id)] = tf.keras.layers.Input(shape=(self.feature_size, ), dtype=tf.int32, sparse=True, name=("unclick" + "_" + "position" + "_"+str(i) + "_" + str(group_id)))
                self.group_feature["feedback" + "_" + "position" + "_" + str(i) + "_" + str(group_id)] = tf.keras.layers.Input(shape=(self.feature_size, ), dtype=tf.int32, sparse=True, name=("feedback" + "_" + "position" + "_"+str(i) + "_" + str(group_id)))
        self.group_feature["clicked_histLen"] = tf.keras.layers.Input(shape=(), dtype=tf.float32, name=("clicked_histLen"))
        self.group_feature["unclick_histLen"] = tf.keras.layers.Input(shape=(), dtype=tf.float32, name=("unclick_histLen"))
        self.group_feature["feedback_histLen"] = tf.keras.layers.Input(shape=(), dtype=tf.float32, name=("feedback_histLen"))

    def embedding_lookup(self, group_ids, prefix=""):
        embeddings = []
        for group_id in group_ids:
            embedding = self.embed_layer(self.group_feature[prefix + str(group_id)])
            embeddings.append(embedding)
        embedding_out = tf.concat(embeddings, axis=1)
        return embedding_out

    def __call__(self,):
        clicked_embeddings = []
        unclick_embeddings = []
        feedback_embeddings = []
        self.embed_layer = Embedding_Lookup(self.feature_size, self.embed_dim,
                                            tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.01),
                                            name="embedding_w")
        # batch_size, len(main_group_ids) * embed_dim, 相同field之间的特征求mean
        main_embedding = self.embedding_lookup(self.main_group_ids, prefix="main_")

        # batch_size, len(candidate_group_ids) * embed_dim, 相同field之间的特征求mean
        candidate_embedding = self.embedding_lookup(self.candidate_group_ids, prefix="candidate_")

        seq_emb = Sequence_Embedding(self.clicked_item_dim, self.pos_item_dim, self.unclick_item_dim, self.feedback_item_dim, self.item_dim)
        for i in range(0, self.hist_size):
            # 一个用户不用field都有30长的序列，对于用户序列，每个序号求相同field的mean
            # batch_size, len(clicked_group_ids) * embed_dim
            clicked_embedding = self.embedding_lookup(self.clicked_group_ids, prefix = "clicked" + "_" + str(i) + "_")
            unclick_embedding = self.embedding_lookup(self.unclick_group_ids, prefix="unclick" + "_" + str(i) + "_")
            feedback_embedding = self.embedding_lookup(self.feedback_group_ids, prefix="feedback" + "_" + str(i) + "_")
            clicked_position_embedding = self.embedding_lookup(self.pos_group_ids, prefix="clicked" + "_" + "position" + "_" + str(i) + "_")
            unclick_position_embedding = self.embedding_lookup(self.pos_group_ids, prefix="unclick" + "_" + "position" + "_" + str(i) + "_")
            feedback_position_embedding = self.embedding_lookup(self.pos_group_ids, prefix="feedback" + "_" + "position" + "_" + str(i) + "_")
            # 位置信息concat
            clicked_pos = tf.concat([clicked_embedding, clicked_position_embedding], axis=1)
            unclick_pos = tf.concat([unclick_embedding, unclick_position_embedding], axis=1)
            feedback_pos = tf.concat([feedback_embedding, feedback_position_embedding], axis=1)
            # 特征和位置embedding
            # batch_size, len(clicked_group_ids) * embed_dim
            clicked_z, unclick_z, feedback_z = seq_emb([clicked_pos, unclick_pos, feedback_pos])
            clicked_embeddings.append(clicked_z)
            unclick_embeddings.append(unclick_z)
            feedback_embeddings.append(feedback_z)

        # wide embedding
        main_embeddings_wide = []
        candidate_embeddings_wide = []
        self.embed_wide = Embedding_Lookup(self.feature_size, 1, tf.keras.initializers.Zeros(), name="embedding_wide")
        for group_id in self.main_group_ids:
            # batch_size, len(main_group_ids) * 1
            embedding_wide = self.embed_wide(self.group_feature["main_" + str(group_id)])
            main_embeddings_wide.append(embedding_wide)
        main_embedding_wide = tf.concat(main_embeddings_wide, axis=1)

        for group_id in self.candidate_group_ids:
            # batch_size, len(candidate_group_ids) * 1
            embedding_wide = self.embed_wide(self.group_feature["candidate_" + str(group_id)])
            candidate_embeddings_wide.append(embedding_wide)
        candidate_embedding_wide = tf.concat(candidate_embeddings_wide, axis=1)

        # batch, hist_embedding_dim
        output_clicked = Transformer(self.hist_size, self.item_dim, prefix="clicked")([candidate_embedding, clicked_embeddings, self.group_feature["clicked_histLen"]])
        output_unclick = Transformer(self.hist_size, self.item_dim, prefix="unclick")([candidate_embedding, unclick_embeddings, self.group_feature["unclick_histLen"]])
        output_feedback = Transformer(self.hist_size, self.item_dim, prefix="feedback")([candidate_embedding, feedback_embeddings, self.group_feature["feedback_histLen"]])

        output_unclick_clicked = Attention(self.hist_size, self.item_dim, prefix="unclick_clicked")([output_clicked, unclick_embeddings, self.group_feature["unclick_histLen"]])
        output_unclick_feedback = Attention(self.hist_size, self.item_dim, prefix="unclick_feedback")([output_feedback, unclick_embeddings, self.group_feature["unclick_histLen"]])

        input_embedding = tf.concat([main_embedding, candidate_embedding, output_clicked, output_unclick, output_feedback, output_unclick_clicked, output_unclick_feedback],axis=1)

        #fm part 这个*6估计相当于multihead一样，切分成多个子空间进行两两交叉组合
        m = len(self.main_group_ids) + len(self.candidate_group_ids) * 6
        fm_in = tf.reshape(input_embedding, shape=[-1, m, self.embed_dim])
        sum1 = tf.reduce_sum(fm_in, axis=1)
        sum2 = tf.reduce_sum(fm_in * fm_in, axis=1)
        fm = (sum1 * sum1 - sum2) * 0.5

        #deep part
        deep = DNN([32, 16])(input_embedding)

        z = tf.concat([deep, fm, main_embedding_wide, candidate_embedding_wide], axis=1)

        results = DNN([1,], activation="sigmoid")(z)

        return tf.reshape(results, [-1, 1])