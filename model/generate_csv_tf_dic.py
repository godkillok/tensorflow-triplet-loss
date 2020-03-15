#!/usr/bin/env python
import tensorflow as tf
import os
import re
import numpy as np
import random

import random
import json
import time
from common_tool import per_line,parse_line_dict,ini
flags = tf.app.flags
flags.DEFINE_string("data_dir", "/data/tanggp/tmp/Starspace/python/test/", "Directory containing the dataset")
flags.DEFINE_string("pad_word", '0', "used for pad sentence")
flags.DEFINE_string("OOV", 'unknown', "used for pad sentence")
flags.DEFINE_string("path_vocab", "/data/tanggp/tmp/Starspace/python/test/textcnn_words.txt", "used for word index")
flags.DEFINE_string("path_label",  os.path.join("/data/tanggp/tmp/Starspace/python/test/", 'textcnn_label_sort'), "Directory containing the dataset")
FLAGS = flags.FLAGS

sentence_max_len = 60
pad_word = FLAGS.pad_word
OOV =FLAGS.OOV
label_class=[]
author_calss=[]
def feature_auto(value):
    if isinstance(value, int):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    elif isinstance(value, list):
        if isinstance(value[0],int):
            try:
                tf.train.Feature(int64_list=tf.train.Int64List(value=value))
            except:
                print(value)
            return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
        elif isinstance(value[0],float):
            try:
                tf.train.Feature(int64_list=tf.train.FloatList(value=value))
            except:
                print(value)
            return tf.train.Feature(int64_list=tf.train.FloatList(value=value))
        else:
            print("list type error")

    elif isinstance(value, str):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))

    elif isinstance(value, float):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))




def per_thouds_lines_dict(result_lines, path_text, count,pad_word,flag_name=''):
    tf_lines = []


    for rl_num,rl in enumerate(result_lines):
        text=rl[0]
        label=rl[1]
        tags=rl[2]
        if len(text) >= sentence_max_len:
            text = text[0: sentence_max_len]
        else:
            text += [pad_word] * (sentence_max_len - len(text))
        tags_=[]
        for tag in tags:
            if len(tag) >= 10:
                tag = tag[0: 10]
            else:
                tag += [pad_word] * (10 - len(tag))
            tags_+=tag
        g={"text":text,"labels":label,"tags":tags_}
        tf_lines.append(g)
        # if rl_num>1 and rl_num%10000==0:
        #     flag_name=str(rl_num)
        #     write_tfrecords(tf_lines, path_text, count)
        #     tf_lines = []
    if len(tf_lines)>0:
        flag_name = str(rl_num)
        print(len(tf_lines))
        #time.sleep(5)
        write_tfrecords(tf_lines, path_text, count)
            # tf_lines=[]

def  parse_line_dict2(line,vocab_dict,label_dict):
    tokens, labels=per_line(line)
    return parse_line_dict(tokens,labels,vocab_dict,label_dict,vocab_dict[OOV])



def generate_tf_dic(path_text,vocab_dict,label_dict):


    result_lines = []
    count = 0
    with open(path_text, 'r', encoding='utf8') as f:
        lines = f.readlines()
        random.shuffle(lines)
        for line in lines:
            count+=1
            result_lines.append(parse_line_dict2(line,vocab_dict,label_dict))
            if count>0 and count % 50000 == 0:
                print(count)
                pad_word=vocab_dict[pad_word]
                per_thouds_lines_dict(result_lines, path_text, count,pad_word)
                result_lines = []
        if len(result_lines)>0:
            per_thouds_lines_dict(result_lines, path_text, count)


def write_tfrecords(tf_lines, path_text, count):
    (root_path, output_filename) = os.path.split(path_text)
    output_filename = output_filename.split('.')[0]
    output_filename='author_text_cnn_'+output_filename
    output_file = output_filename + '_' + str(count)+ '.tfrecords'

    print("Start to convert {} to {}".format(len(tf_lines), os.path.join(root_path, output_file)))

    writer = tf.python_io.TFRecordWriter(os.path.join(root_path, output_file))
    random.shuffle(tf_lines)
    num = 0
    for i,data in enumerate(tf_lines):
        if i==0:
            print("Start to convert {}".format(data))

        #{"text":text,"labels":label,"tags":tags_}
        text = data["text"]
        labels = data["labels"]
        tags=data["tags"]
        example = tf.train.Example(features=tf.train.Features(feature={
            'text': feature_auto(list(text)),
            'labels': feature_auto(list(labels)),
            'tags': feature_auto(list(tags))
        }))

        writer.write(example.SerializeToString())
        num += 1
        # if num % 1000 == 0:
        #     output_file = output_filename + '_' + str(count) + '_' + str(num)+'_' + flag_name + '.tfrecords'
        #     writer = tf.python_io.TFRecordWriter(os.path.join(root_path, output_file))
        #     print("Start convert to {}".format(output_file))


def main():

    vocab_dict, label_dict= ini(FLAGS.path_vocab,FLAGS.path_label,pad_word,OOV)

    generate_tf_dic(os.path.join(FLAGS.data_dir, 'tag_space'),vocab_dict,label_dict)
    # generate_tf_dic(os.path.join(FLAGS.data_dir, 'tag_space2'),vocab_dict,label_dict)
    # generate_tf_dic(os.path.join(FLAGS.data_dir, 'txt_valid'), vocab_dict,label_dict)
    # s3_input = FLAGS.data_dir
    # for root, dirs, files in os.walk(s3_input):
    #     for file in files:
    #         # if file.endswith("ain_set.csv"):
    #         #     print('start to process file {}'.format(file))
    #         generate_tf_dic(os.path.join(root, file))
    # os.system('cd {}'.format(s3_input))
    # os.system('find . -name "*" -type f -size 0c | xargs -n 1 rm -f')


if __name__ == "__main__":
    main()
