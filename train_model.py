# -*- coding:utf-8 -*-
import tensorflow as tf
from pathlib import Path
import numpy as np
import scipy as sp
from model import DFN

batch_size = 960
hist_size = 30
data_dict = {}
feed_dict = {}
batch_idx = 0
feature_size = 1048573
epoch = 25

def data_set(data_dict, feature, string):
        if string not in data_dict:
             data_dict[string] =[[feature]]
        else:
             if(len(data_dict[string]) < batch_idx + 1):
                 data_dict[string].append([feature])
             else:
                 data_dict[string][batch_idx].append(feature)

def input_data_set(data_dict, features, prefix=""):
    global main_group_ids, candidate_group_ids
    for feature in features:
        feature = feature.split(":")
        feature = int(feature[0])
        group_id = feature >> 48
        feature = feature % feature_size
        if prefix == "main_":
            if group_id not in main_group_ids:
                continue
        elif prefix == "candidate_":
            if group_id not in candidate_group_ids:
                continue
        data_set(data_dict, feature, prefix+str(group_id))

def input_hist_data_set(data_dict, hist_features, hist_group_ids, pos_group_ids, hist_size, prefix=""):
    hist_len = len(hist_features)
    if hist_features[0] == '\n' or hist_features[0] == '' or hist_features[0] == ' ':
          hist_len = 0
    for i in range(0, hist_size):
        if i < hist_len:
            features = hist_features[i].split()
            for feature in features:
                 feature = feature.split(":")
                 feature = int(feature[0])
                 group_id = feature >> 48
                 feature = feature % feature_size
                 if group_id in pos_group_ids:
                       data_set(data_dict, feature, prefix+"position_"+str(i)+"_"+str(group_id))
                 else:
                       data_set(data_dict, feature, prefix+str(i)+"_"+str(group_id))
        else:
            for group_id in hist_group_ids:
                 data_set(data_dict, 0, prefix+str(i)+"_"+str(group_id))
            for group_id in pos_group_ids:
                 data_set(data_dict, 0, prefix+"position_"+str(i)+"_"+str(group_id))
             
    if prefix+"histLen" not in data_dict:
            data_dict[prefix+"histLen"] = [hist_len]
    else:
            data_dict[prefix+"histLen"].append(hist_len)

def data_dict_sparse_feature(data_dict, string, dtype):
    index, value = [], []
#     rows, cols, value = [], [], []
    for i in range(batch_size):
           for k in range(len(data_dict[string][i])):
#                 rows.append(i)
#                 cols.append(k)
                index.append(np.array([i, k], dtype = np.int32))
                value.append(data_dict[string][i][k])
#     iv = sp.sparse.coo_matrix((value, (rows, cols)), shape=[len(data_dict[string]), feature_size])
#     if dtype == tf.int32:
#         iv = iv.astype(np.int32)
#     elif dtype == tf.float32:
#         iv = iv.astype(np.float32)
    iv = tf.sparse.SparseTensor(index, value, [len(data_dict[string]), feature_size])
    iv = tf.cast(iv, dtype=dtype)
    data_dict[string] = iv


def train_data_process(data, data_dict, main_group_ids, candidate_group_ids, clicked_group_ids, unclick_group_ids, feedback_group_ids, pos_group_ids):
    data = data.split('\t')
    label = float(data[0])
    weight = float(data[1])
    features = data[2].split('|')
    main_features = features[0].split()
    candidate_features = features[1].split()
    clicked_features = features[2].split(';')
    unclick_features = features[3].split(';')
    feedback_features = features[4].split(';')
    if "label" not in data_dict:
        data_dict["label"] = [label]
    else:
        data_dict["label"].append(label)
    
    if "weight" not in data_dict:
        data_dict["weight"] = [weight]
    else:
        data_dict["weight"].append(weight)
    
    input_data_set(data_dict, main_features, "main_")
    input_data_set(data_dict, candidate_features, "candidate_")
    input_hist_data_set(data_dict, clicked_features, clicked_group_ids, pos_group_ids, hist_size, "clicked_")
    input_hist_data_set(data_dict, unclick_features, unclick_group_ids, pos_group_ids, hist_size, "unclick_")
    input_hist_data_set(data_dict, feedback_features, feedback_group_ids, pos_group_ids, hist_size, "feedback_")


def data_gen(path):
    global batch_idx, data_dict, batch_size, main_group_ids, candidate_group_ids, clicked_group_ids, unclick_group_ids, feedback_group_ids, pos_group_ids
    while True:
        f = path.open(mode='r')
        line = f.readline()
        while line:
            train_data_process(line, data_dict, main_group_ids, candidate_group_ids, clicked_group_ids, unclick_group_ids, feedback_group_ids, pos_group_ids)
            if batch_idx < batch_size -1: 
                batch_idx += 1
            else:
                for group_id in main_group_ids:
                    data_name = "main_" + str(group_id)
                    data_dict_sparse_feature(data_dict, data_name, tf.int32)
                for group_id in candidate_group_ids:
                    data_name = "candidate_" + str(group_id)
                    data_dict_sparse_feature(data_dict, data_name, tf.int32)
                for i in range(hist_size):
                    for group_id in clicked_group_ids:
                        data_name = "clicked_" + str(i) + "_" + str(group_id)
                        data_dict_sparse_feature(data_dict, data_name, tf.int32) 
                    for group_id in unclick_group_ids:
                        data_name = "unclick_" + str(i) + "_" + str(group_id)
                        data_dict_sparse_feature(data_dict, data_name, tf.int32) 
                    for group_id in feedback_group_ids:
                        data_name = "feedback_" + str(i) + "_" + str(group_id)
                        data_dict_sparse_feature(data_dict, data_name, tf.int32)
                    for group_id in pos_group_ids:   
                        data_name = "clicked_position_" + str(i) + "_" + str(group_id)
                        data_dict_sparse_feature(data_dict, data_name, tf.int32)
                        data_name = "unclick_position_" + str(i) + "_" + str(group_id)
                        data_dict_sparse_feature(data_dict, data_name, tf.int32)
                        data_name = "feedback_position_" + str(i) + "_" + str(group_id)
                        data_dict_sparse_feature(data_dict, data_name, tf.int32)
                data_dict["clicked_histLen"] = tf.convert_to_tensor(data_dict["clicked_histLen"], dtype=tf.float32)
                data_dict["unclick_histLen"] = tf.convert_to_tensor(data_dict["unclick_histLen"], dtype=tf.float32)
                data_dict["feedback_histLen"] = tf.convert_to_tensor(data_dict["feedback_histLen"], dtype=tf.float32)
                data_dict["label"] = tf.convert_to_tensor(data_dict["label"], dtype=tf.float32)
                data_dict["weight"] = tf.convert_to_tensor(data_dict["weight"], dtype=tf.float32)
                data_input = {k: v for k, v in data_dict.items() if k != "label" and k != "weight"}
                labels = data_dict["label"]
                weights = data_dict["weight"]
                batch_idx = 0
                data_dict = {}
                yield (data_input, labels, weights)
            line = f.readline()
        f.close()

if __name__ == "__main__":
    main_group_ids=[16,10001,10002,10003,21,10006,10019,10034,20147,20148,10035,20156,
                    61,10047,10048,10049,10050,10055,10056,60]
    candidate_group_ids=[3060,3061,3062,3063,3064]
    clicked_group_ids=[3060,3061,3062,3063,3064]
    unclick_group_ids=[3060,3061,3062,3063,3064]
    feedback_group_ids=[3060,3061,3063,3064]
    pos_group_ids=[3065]

    path = Path(r"E:\ML_study\deepctr\dfn_tf2\example")
    train_data, train_label, sample_weight = next(data_gen(path))
    dfn = DFN(main_group_ids, candidate_group_ids, clicked_group_ids, unclick_group_ids, feedback_group_ids, pos_group_ids)
    output = dfn()
    model = tf.keras.models.Model(inputs=dfn.group_feature, outputs=output) 
    # print(model)
    # model.compile(tf.keras.optimizers.Adagrad(), "binary_crossentropy",
    #               metrics=['binary_crossentropy'], )
    # history = model.fit(train_data, train_label, epochs=25, batch_size=256, shuffle=True, sample_weight=sample_weight)
    # history = model.fit(data_gen(path), epochs=25, batch_size=batch_size, shuffle=False)
                