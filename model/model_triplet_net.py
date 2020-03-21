"""Define functions to create the triplet loss with online triplet mining."""

import tensorflow as tf
import numpy  as np
from model.triplet_loss import batch_all_triplet_loss
from model.triplet_loss import batch_hard_triplet_loss
text_max=60
tag_max=15
def random_label(labels_lists):
    rows = tf.shape(labels_lists)[0]
    # Number of zeros on each row
    zero_mask = tf.cast(tf.greater(labels_lists, 0), tf.int32)
    num_zeros = tf.reduce_sum(zero_mask, axis=1)
    # Random values
    r = tf.random_uniform([rows], 0, 1, dtype=tf.float32)
    # Multiply by the number of zeros to decide which of the zeros you pick
    zero_idx = tf.cast(tf.floor(r * tf.cast(num_zeros, r.dtype)), tf.int32)
    # Find the indices of the smallest values, which should be the zeros
    _, zero_pos = tf.nn.top_k(labels_lists, k=tf.maximum(tf.reduce_max(num_zeros), 1))
    # Select the corresponding position of each row
    result = tf.gather_nd(zero_pos, tf.stack([tf.range(rows), zero_idx], axis=1))
    return result

def random_tag(labels_lists,tags_list):
    print("tags_list {}".format(tags_list))
    labels=random_label(labels_lists)
    rows = tf.shape(labels_lists)[0]
    result=tf.gather_nd(tags_list, tf.stack([tf.range(rows), labels], axis=1))
    return result,labels

def assign_pretrained_word_embedding(params):
    # print("using pre-trained word emebedding.started.word2vec_model_path:",fast_text)
    # import fastText as ft
    # word2vec_model = ft.load_model(FLAGS.fast_text)

    vocab_size=params["vocab_size"]
    embedding_size=100
    word_embedding_final = np.zeros((vocab_size,embedding_size))  # create an empty word_embedding list.

    bound = np.sqrt(6.0) / np.sqrt(vocab_size)  # bound for random variables.
    count_exist = 0
    count_not_exist = 0
    path_vocab=""
    with open(path_vocab, 'r', encoding='utf8') as f:
        lines = f.readlines()
        vocab = {l.strip(): i for i, l in enumerate(lines)}
    # print(len(vocab))
    # print(len(lines))
    # print(vocab_size)
    # #assert len(vocab)==vocab_size

    for (word, idx) in vocab.items():
        # embedding=word2vec_model.get_word_vector(word)
        # if embedding is not None:  # the 'word' exist a embedding
        #     word_embedding_final[idx,:] = embedding
        #     count_exist = count_exist + 1  # assign array to this word.
        # else:  # no embedding for this word
        word_embedding_final[idx, :] = np.random.uniform(-bound, bound, embedding_size);
        count_not_exist = count_not_exist + 1  # init a random value for the word.

    # word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)  # convert to tensor

    print("word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)
    print("using pre-trained word emebedding.ended...")
    return word_embedding_final

def cnn(sentence,embeddings,num_filters,filter_sizes,sentence_max_len,mode):
    #print("sentence.shape[2] {}".format(sentence.shape))
    sentence = tf.nn.embedding_lookup(embeddings, sentence)
    sentence = tf.expand_dims(sentence, -1)
    pooled_outputs = []
    #print("sentence.shape[2] {}".format(sentence.shape))
    filter_len=sentence.shape[2]
    for filter_size in filter_sizes:

        conv = tf.layers.conv2d(
            sentence,
            filters=num_filters,
            kernel_size=[filter_size, filter_len],
            strides=(1, 1),
            padding="VALID",
          activation=tf.nn.relu
        )
        conv = tf.layers.batch_normalization(conv, training=(mode == tf.estimator.ModeKeys.TRAIN))
        # conv = tf.nn.relu(conv)

        # conv = tf.layers.conv2d(
        #     conv,
        #     filters=FLAGS.num_filters,
        #     kernel_size=[filter_size, 1],
        #     strides=(1, 1),
        #     padding="SAME"
        # )#activation=tf.nn.relu
        # conv = tf.nn.relu(conv)
        # b = tf.get_variable("b-%s" % filter_size, [FLAGS.num_filters])  # ADD 2017-06-09
        # if 'dropout_rate' in params and params['dropout_rate'] > 0.0:
        #     # h_pool_flat = tf.layers.batch_normalization(h_pool_flat, training=(mode == tf.estimator.ModeKeys.TRAIN))
        #
        #     conv = tf.layers.dropout(conv, params['dropout_rate'],
        #                              training=(mode == tf.estimator.ModeKeys.TRAIN))
        # # conv = tf.layers.batch_normalization(conv, training=(mode == tf.estimator.ModeKeys.TRAIN))

        pool = tf.layers.max_pooling2d(
            conv,
            pool_size=[sentence_max_len - filter_size + 1, 1],
            strides=(1, 1),
            padding="VALID")

        pooled_outputs.append(pool)

    h_pool = tf.concat(pooled_outputs, 3)  # shape: (batch, 1, len(filter_size) * embedding_size, 1)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters * len(filter_sizes)])  # shape: (batch, len(filter_size) * embedding_size)
    logits = tf.layers.dense(h_pool_flat, 100, activation=None)
    return logits

def  get_tag_embedding(labels_lists,y_tower,word_embedding,mode):
    tags,labels=random_tag(labels_lists,y_tower)
    print("tag {}".format(tags.shape))
    num_filters=3
    filter_sizes=[2,3,4]
    sentence_max_len=tag_max
    tag_logit=cnn(tags, word_embedding, num_filters, filter_sizes, sentence_max_len,mode)
    return tag_logit,labels

def get_txt_embedding(x_tower,word_embedding,mode):
    sentence_max_len=60
    num_filters=3
    filter_sizes=[2,3,4,5,7]
    # sentence_max_len=10
    sentence_logit=cnn(x_tower, word_embedding, num_filters, filter_sizes, sentence_max_len,mode)
    return sentence_logit

def model_fn(features, mode,params):
    vocab_size=400010
    embedding_size=100
    word_embedding = tf.get_variable(name="embeddings", dtype=tf.float32,
                                 shape=[vocab_size , embedding_size],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))

    y_tower=features["tags"]
    y_tower=tf.reshape(y_tower,[-1,12,tag_max])
    labels_lists=features["labels"]
    x_tower=features["text"]
    tag_logit,labels= get_tag_embedding(labels_lists,y_tower,word_embedding,mode)
    sentence_logit=get_txt_embedding(x_tower, word_embedding,mode)
    sentence_logit = tf.nn.l2_normalize(sentence_logit, dim=1)
    tag_logit = tf.nn.l2_normalize(tag_logit, dim=1)
    embedding_mean_norm = tf.reduce_mean(tf.norm(sentence_logit, axis=1))
    tf.summary.scalar("embedding_mean_norm", embedding_mean_norm)



        # return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    labels = tf.cast(labels, tf.int64)  # Tensor("Cast:0", shape=(?,), dtype=int64)
    triplet_strategy = "batch_all"
    # Define triplet loss
    if triplet_strategy == "batch_all":
        loss, fraction,num_positive_triplets,cosine,neg,pairwise_dist = batch_all_triplet_loss(labels, sentence_logit,tag_logit, margin=0.05,
                                                squared=False)


        tf.summary.scalar('loss1', num_positive_triplets)
        tf.summary.scalar('fraction_positive_triplets', fraction)

    else : #triplet_strategy == "batch_hard"
        loss = batch_hard_triplet_loss(labels, sentence_logit,tag_logit, margin=0.05,
                                       squared=False)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'sentence_logit': sentence_logit,"tag_logit":tag_logit,"labels":labels,"cosine":cosine,"neg":neg,"pairwise_dist":pairwise_dist}
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions, export_outputs=export_outputs)
    # else:
    #     raise ValueError("Triplet strategy not recognized: {}".format(params.triplet_strategy))

    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    # TODO: some other metrics like rank-1 accuracy?
    # with tf.variable_scope("metrics"):
    #     eval_metric_ops = {"embedding_mean_norm": tf.metrics.mean(embedding_mean_norm)}
    eval_metric_ops = {"cosine": tf.metrics.mean(cosine),"neg":tf.metrics.mean(neg)}

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)

    # Summaries for training
    tf.summary.scalar('loss', loss)



    # tf.summary.image('train_image', images, max_outputs=1)

    # Define training step that minimizes the loss with the Adam optimizer
    optimizer = tf.train.AdamOptimizer(0.1)
    global_step = tf.train.get_global_step()
    # if params.use_batch_norm:
    #     # Add a dependency to update the moving mean and variance for batch normalization
    #     with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    #         train_op = optimizer.minimize(loss, global_step=global_step)
    # else:
    train_op = optimizer.minimize(loss, global_step=global_step)

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)