# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from collections import OrderedDict

from tensorflow.python.ops.array_ops import zeros

class DFN(object):

    def __init__(self, main_group_ids, candidate_group_ids, clicked_group_ids, unclick_group_ids, feedback_group_ids, pos_group_ids, batch_size=256, embed_dim=16, feature_size=1048573, hist_size=30):
        self._embed_dim = embed_dim
        self._feature_size = feature_size
        self._hist_size = hist_size
        self._batch_size = batch_size
        self._group_feature = OrderedDict()
        self._clicked_item_dim = len(clicked_group_ids) * embed_dim
        self._unclick_item_dim = len(unclick_group_ids) * embed_dim
        self._feedback_item_dim = len(feedback_group_ids) * embed_dim
        self._item_dim = self._clicked_item_dim
        self._pos_item_dim = len(pos_group_ids) * embed_dim
        self._results = None
        # build input
        for group_id in main_group_ids:
          self._group_feature["main_" + str(group_id)] = tf.keras.layers.Input(shape=(self._feature_size, ), dtype=tf.int32, sparse=True, name=("main_" + str(group_id)))
        for group_id in candidate_group_ids:
          self._group_feature["candidate_" + str(group_id)] = tf.keras.layers.Input(shape=(self._feature_size, ), dtype=tf.int32, sparse=True, name=("candidate_" + str(group_id)))
 
        for i in range(0, hist_size):
            for group_id in clicked_group_ids:
                self._group_feature["clicked" + "_" + str(i) + "_" + str(group_id)] = tf.keras.layers.Input(shape=(self._feature_size, ), dtype=tf.int32, sparse=True, name=("clicked" + "_" + str(i) + "_" + str(group_id)))
            for group_id in unclick_group_ids:
                self._group_feature["unclick" + "_" + str(i) + "_" + str(group_id)] = tf.keras.layers.Input(shape=(self._feature_size, ), dtype=tf.int32, sparse=True, name=("unclick" + "_" + str(i) + "_" + str(group_id)))
            for group_id in feedback_group_ids:
                self._group_feature["feedback" + "_" + str(i) + "_" + str(group_id)] = tf.keras.layers.Input(shape=(self._feature_size, ), dtype=tf.int32, sparse=True, name=("feedback" + "_" + str(i)+"_"+str(group_id)))  
            for group_id in pos_group_ids:
                self._group_feature["clicked" + "_" + "position" + "_" + str(i) + "_" + str(group_id)] = tf.keras.layers.Input(shape=(self._feature_size, ), dtype=tf.int32, sparse=True, name=("clicked" + "_" + "position" + "_"+str(i) + "_"+str(group_id)))
                self._group_feature["unclick" + "_" + "position" + "_" + str(i) + "_" + str(group_id)] = tf.keras.layers.Input(shape=(self._feature_size, ), dtype=tf.int32, sparse=True, name=("unclick" + "_" + "position" + "_"+str(i) + "_" + str(group_id)))
                self._group_feature["feedback" + "_" + "position" + "_" + str(i) + "_" + str(group_id)] = tf.keras.layers.Input(shape=(self._feature_size, ), dtype=tf.int32, sparse=True, name=("feedback" + "_" + "position" + "_"+str(i) + "_" + str(group_id)))
        self._group_feature["clicked_histLen"] = tf.keras.layers.Input(shape=(1, ), dtype=tf.float32, name=("clicked_histLen"))
        self._group_feature["unclick_histLen"] = tf.keras.layers.Input(shape=(1, ), dtype=tf.float32, name=("unclick_histLen"))
        self._group_feature["feedback_histLen"] = tf.keras.layers.Input(shape=(1, ), dtype=tf.float32, name=("feedback_histLen"))
        # self.weights = tf.keras.layers.Input(shape=(1, ), dtype=tf.float32, name='weights')
        # self.labels = tf.keras.layers.Input(shape=(1, ), dtype=tf.float32, name='label')
        self.buildDFN(main_group_ids, candidate_group_ids, clicked_group_ids, unclick_group_ids, feedback_group_ids, pos_group_ids)

    def embedding_lookup(self, embedding_w, group_ids, prefix=""):
        embeddings = []
        for group_id in group_ids:
            embedding = tf.nn.embedding_lookup_sparse(embedding_w, self._group_feature[prefix + str(group_id)], sp_weights=None, combiner='mean')
            embeddings.append(embedding)
        embedding_out = tf.concat(embeddings, axis=1)
        return embedding_out


    @property
    def group_feature(self):
        return self._group_feature

    def buildDFN(self, main_group_ids, candidate_group_ids, clicked_group_ids, unclick_group_ids, feedback_group_ids, pos_group_ids):
    #embedding 
        clicked_embeddings = []
        unclick_embeddings = []
        feedback_embeddings = []
        init_w = tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.01)
        embed_w = tf.Variable(init_w(shape=[self._feature_size, self._embed_dim]), name='embedding_w', dtype=tf.float32)
        self.embed_w = embed_w
        # batch_size, len(main_group_ids) * embed_dim, 相同field之间的特征求mean
        main_embedding = self.embedding_lookup(embed_w, main_group_ids, prefix="main_")

        # batch_size, len(candidate_group_ids) * embed_dim, 相同field之间的特征求mean
        candidate_embedding = self.embedding_lookup(embed_w, candidate_group_ids, prefix="candidate_")
        
        init_seq = tf.keras.initializers.GlorotNormal()
        pos_w_clicked = tf.Variable(init_seq(shape=[self._clicked_item_dim + self._pos_item_dim, self._item_dim]), dtype=tf.float32, name="pos_w_clicked")
        pos_w_unclick = tf.Variable(init_seq(shape=[self._unclick_item_dim + self._pos_item_dim, self._item_dim]), dtype=tf.float32, name="pos_w_unclick")
        pos_w_feedback = tf.Variable(init_seq(shape=[self._feedback_item_dim + self._pos_item_dim, self._item_dim]), dtype=tf.float32, name="pos_w_feedback")

        for i in range(0, self._hist_size):
            # 一个用户不用field都有30长的序列，对于用户序列，每个序号求相同field的mean
            # batch_size, len(clicked_group_ids) * embed_dim
            clicked_embedding = self.embedding_lookup(embed_w, clicked_group_ids, prefix = "clicked" + "_" + str(i) + "_")
            unclick_embedding = self.embedding_lookup(embed_w, unclick_group_ids, prefix="unclick" + "_" + str(i) + "_")
            feedback_embedding = self.embedding_lookup(embed_w, feedback_group_ids, prefix="feedback" + "_" + str(i) + "_")
            clicked_position_embedding = self.embedding_lookup(embed_w, pos_group_ids, prefix="clicked" + "_" + "position" + "_" + str(i) + "_")
            unclick_position_embedding = self.embedding_lookup(embed_w, pos_group_ids, prefix="unclick" + "_" + "position" + "_" + str(i) + "_")
            feedback_position_embedding = self.embedding_lookup(embed_w, pos_group_ids, prefix="feedback" + "_" + "position" + "_" + str(i) + "_")
            # 位置信息concat
            clicked_pos = tf.concat([clicked_embedding, clicked_position_embedding], axis=1)
            unclick_pos = tf.concat([unclick_embedding, unclick_position_embedding], axis=1)
            feedback_pos = tf.concat([feedback_embedding, feedback_position_embedding], axis=1)
            # 特征和位置embedding
            # batch_size, len(clicked_group_ids) * embed_dim
            clicked_z = tf.matmul(clicked_pos, pos_w_clicked)
            unclick_z = tf.matmul(unclick_pos, pos_w_unclick)
            feedback_z = tf.matmul(feedback_pos, pos_w_feedback)
            clicked_embeddings.append(clicked_z)
            unclick_embeddings.append(unclick_z)
            feedback_embeddings.append(feedback_z)

        # wide embedding
        main_embeddings_wide = []
        candidate_embeddings_wide = []
        embed_wide = tf.Variable(tf.zeros([self._feature_size, 1]), name="embedding_wide")
        for group_id in main_group_ids:
            # batch_size, len(main_group_ids) * 1
            embedding_wide = tf.nn.embedding_lookup_sparse(embed_wide, self._group_feature["main_" + str(group_id)], sp_weights=None, combiner='mean')
            main_embeddings_wide.append(embedding_wide)
        main_embedding_wide = tf.concat(main_embeddings_wide, axis=1)

        for group_id in candidate_group_ids:
            # batch_size, len(candidate_group_ids) * 1
           embedding_wide = tf.nn.embedding_lookup_sparse(embed_wide, self._group_feature["candidate_" + str(group_id)], sp_weights=None, combiner='mean')
           candidate_embeddings_wide.append(embedding_wide)
        candidate_embedding_wide = tf.concat(candidate_embeddings_wide, axis=1)

        # batch, hist_embedding_dim
        output_clicked = self.transformer(candidate_embedding, clicked_embeddings, self._item_dim, self._group_feature["clicked_histLen"], prefix="clicked")
        output_unclick = self.transformer(candidate_embedding, unclick_embeddings, self._item_dim, self._group_feature["unclick_histLen"], prefix="unclick")
        output_feedback = self.transformer(candidate_embedding, feedback_embeddings, self._item_dim, self._group_feature["feedback_histLen"], prefix="feedback")
        output_unclick_clicked = self.attention(output_clicked, unclick_embeddings, self._item_dim, self._group_feature["unclick_histLen"], prefix="unclick_clicked")
        output_unclick_feedback = self.attention(output_feedback, unclick_embeddings, self._item_dim, self._group_feature["unclick_histLen"], prefix="unclick_feedback")

        input_embedding = tf.concat([main_embedding, candidate_embedding, output_clicked, output_unclick, output_feedback, output_unclick_clicked, output_unclick_feedback],axis=1)

        #fm part 这个*6估计相当于multihead一样，切分成多个子空间进行两两交叉组合
        m = len(main_group_ids) + len(candidate_group_ids) * 6
        fm_in = tf.reshape(input_embedding, shape=[-1, m, self._embed_dim])
        fm = self.batch_group_fm_quadratic2(fm_in)

        #deep part
        deep = self.stacked_fully_connect(input_embedding, [32, 16])

        z = tf.concat([deep, fm, main_embedding_wide, candidate_embedding_wide], axis=1)
        self._results = self.stacked_fully_connect(z, [1], "sigmoid", prefix="output")
        self._results = tf.reshape(self._results, [-1, 1])


    def attention(self, candidate_embedding, hist_embeddings, hist_embedding_dim, hisLens, prefix=""):
        attention_hidden_ = 32
        attW1 = tf.Variable(tf.keras.initializers.GlorotNormal()(shape=[hist_embedding_dim * 4, attention_hidden_]), name=prefix + "attention_hidden_w1", dtype=tf.float32)
        attB1 = tf.Variable(tf.zeros(shape=[attention_hidden_]), name=prefix + "attention_hidden_b1", dtype=tf.float32)

        attW2 = tf.Variable(tf.keras.initializers.GlorotNormal()(shape=[attention_hidden_, 1]), name=prefix + "attention_hidden_w2", dtype=tf.float32)
        attB2 = tf.Variable(tf.zeros(shape=[1]), name=prefix + "attention_hidden_b2", dtype=tf.float32)
        
        hist_embedding_list=[]
        for i in range(0, self._hist_size):
            # batch, hist_embedding_dim * 4
            z1 = tf.concat([candidate_embedding, hist_embeddings[i], candidate_embedding*hist_embeddings[i], candidate_embedding-hist_embeddings[i]], axis=1)
            hist_embedding_list.append(z1)
        hist_z_all = tf.stack(hist_embeddings, axis=1) #(batch, hist_size, hist_embedding_dim)
        z2 = tf.concat(hist_embedding_list, axis=1)  #(batch, hist_size * hist_embedding_dim * 4)
        z3 = tf.reshape(z2, [-1, self._hist_size, 4 * hist_embedding_dim])
        z4 = tf.tensordot(z3, attW1, axes=1) + attB1 #(batch , hist_size, attention_hidden_)
        z5 = tf.nn.relu(z4)
        z6 = tf.tensordot(z5, attW2, axes=1) + attB2 #(batch, hist_size, 1)
        att_w_all = tf.reshape(z6, [-1, self._hist_size])

        #mask
        hist_masks = tf.sequence_mask(hisLens, self._hist_size) #(batch, hist_size)
        padding = tf.ones_like(att_w_all) * (-2**32 + 1)
        att_w_all_rep = tf.where(hist_masks, att_w_all, padding)

        #scale
        att_w_all_scale = att_w_all_rep / (hist_embedding_dim**0.5)

        #norm
        att_w_all_norm = tf.nn.softmax(att_w_all_scale)

        att_w_all_mul = tf.reshape(att_w_all_norm, [-1, 1, self._hist_size])
        weighted_hist_all = tf.matmul(att_w_all_mul, hist_z_all) #(batch, 1, hist_embedding_dim)
        return tf.reshape(weighted_hist_all, [-1, hist_embedding_dim])
    
    def transformer(self, candidate_embedding, hist_embeddings, hist_embedding_dim, hisLens, prefix=""):
        hist_size = self._hist_size + 1
        hist_z = [candidate_embedding]
        for i in range(0,len(hist_embeddings)):
            hist_z.append(hist_embeddings[i])
        hist_z_all = tf.stack(hist_z, axis=1) #(batch, hist_size, hist_embedding_dim)
        
        headnum = 4
        mutil_head_att = []
        #attention
        for i in range(0, headnum):
            attQ_w = tf.Variable(tf.keras.initializers.GlorotNormal()(shape=[hist_embedding_dim, int(hist_embedding_dim / headnum)]), name=prefix + "attQ_w" + str(i), dtype=tf.float32)
            attK_w = tf.Variable(tf.keras.initializers.GlorotNormal()(shape=[hist_embedding_dim, int(hist_embedding_dim / headnum)]), name=prefix + "attK_w" + str(i), dtype=tf.float32)
            attV_w = tf.Variable(tf.keras.initializers.GlorotNormal()(shape=[hist_embedding_dim, int(hist_embedding_dim / headnum)]), name=prefix + "attV_w" + str(i), dtype=tf.float32)
            
            attQ = tf.tensordot(hist_z_all, attQ_w, axes=1) #(batch, hist_size, hist_embedding_dim/headnum)
            attK = tf.tensordot(hist_z_all, attK_w, axes=1) #(batch, hist_size, hist_embedding_dim/headnum)
            attV = tf.tensordot(hist_z_all, attV_w, axes=1) #(batch, hist_size, hist_embedding_dim/headnum)
            
            attQK = tf.matmul(attQ, attK, transpose_b=True) #(batch, hist_size, hist_size)

            #scale
            attQK_scale = attQK / (hist_embedding_dim ** 0.5)
            padding = tf.ones_like(attQK_scale) * (-2 ** 32 + 1) #(batch, hist_size, hist_size)

            #mask
            key_masks = tf.sequence_mask(hisLens + 1, hist_size)  # (batch, hist_size)
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
        FFN_w0 = tf.Variable(tf.keras.initializers.GlorotNormal()(shape=[hist_embedding_dim, hist_embedding_dim * 4]), name=prefix+'FFN_w0', dtype=tf.float32)
        FFN_b0 = tf.Variable(tf.keras.initializers.GlorotNormal()(shape=[hist_embedding_dim * 4]), name=prefix + 'FFN_b0', dtype=tf.float32)
        
        FFN_w1 = tf.Variable(tf.keras.initializers.GlorotNormal()(shape=[hist_embedding_dim * 4, hist_embedding_dim]), name=prefix+'FFN_w1', dtype=tf.float32)
        FFN_b1 = tf.Variable(tf.keras.initializers.GlorotNormal()(shape=[hist_embedding_dim]), name=prefix+'FFN_b1', dtype=tf.float32)
        
        TH0 = tf.tensordot(outputs_QKV, FFN_w0, axes=1) + FFN_b0 #(batch, hist_size, hist_embedding_dim * 4)
        TZ0 = tf.nn.relu(TH0)
        TH1 = tf.tensordot(TZ0, FFN_w1, axes=1) + FFN_b1
        # average pool
        return tf.reduce_sum(TH1, axis=1) #(batch, hist_embedding_dim)

    def batch_group_fm_quadratic2(self, fm_input):
        assert len(fm_input.shape) == 3
        sum1 = tf.reduce_sum(fm_input, axis=1)
        sum2 = tf.reduce_sum(fm_input * fm_input, axis=1)
        z = (sum1 * sum1 - sum2) * 0.5
        return z 
    
    def stacked_fully_connect(self, x, dims, activation='relu', prefix='deep'):
        activation_dict = {
            "relu": tf.nn.relu,
            "sigmoid": tf.nn.sigmoid,
            "tanh": tf.nn.tanh,
        }   
        assert len(x.shape) == 2
        if dims[0] != x.shape[1]:
            dims = [x.shape[1]] + dims
        dim_size = len(dims) - 1
        hidden = x
        for i in range(0, dim_size):
            w = tf.Variable(tf.keras.initializers.GlorotNormal()(shape=[dims[i], dims[i + 1]]), name=prefix + 'w' + str(i), dtype=tf.float32)
            b = tf.Variable(tf.zeros(shape=[dims[i + 1], ]), name=prefix + 'b' + str(i), dtype=tf.float32)
            hidden = tf.matmul(hidden, w) + b
            if prefix != 'output':
                hidden = activation_dict[activation](hidden)
            else:
                hidden = activation_dict["sigmoid"](hidden)
        return hidden