"""Export model as a saved_model"""


from pathlib import Path
import json
from model.model_triplet_net import model_fn

import tensorflow as tf


import os
# DATADIR = '../../data/example'
# PARAMS = './results/params.json'
MODELDIR = 'C:\\work\\tensorflow-triplet-loss\experiments\\batch_all'

# try:
#     os.makedirs(MODELDIR)
# except:
#     pass
def serving_input_receiver_fn():
    """Serving input_fn that builds features from placeholders

    Returns
    -------
    tf.estimator.export.ServingInputReceiver
    """
    # words = tf.placeholder(dtype=tf.string, shape=[None, None], name='words')
    # nwords = tf.placeholder(dtype=tf.int32, shape=[None], name='nwords')
    # # receiver_tensors = {'words': words, 'nwords': nwords}
    # features = {'words': words, 'nword1s': nwords}
    name_to_features_ = {
        "text": tf.placeholder(dtype=tf.int64, shape=[None, 60]) ,# tf.FixedLenFeature([seq_length], tf.int64),
        "labels": tf.placeholder(dtype=tf.int64, shape=[None, 12]),
        "tags": tf.placeholder(dtype=tf.int64, shape=[None, 120])
    }
    return tf.estimator.export.ServingInputReceiver(name_to_features_, name_to_features_)


if __name__ == '__main__':
    # with Path(PARAMS).open() as f:
    #     params = json.load(f)

    # params['words'] = str(Path(DATADIR, 'vocab.words.txt'))
    # params['chars'] = str(Path(DATADIR, 'vocab.chars.txt'))
    # params['tags'] = str(Path(DATADIR, 'vocab.tags.txt'))
    # params['glove'] = str1(Path(DATADIR, 'glove.npz'))

    num_train_steps = None
    num_warmup_steps = None

    config = tf.estimator.RunConfig(tf_random_seed=230,
                                    model_dir=MODELDIR,
                                    save_summary_steps=5)

    estimator = tf.estimator.Estimator(model_fn, params=None, config=config)

    #estimator.train(lambda: train_input_fn(args.data_dir, params))

    estimator.export_savedmodel(os.path.join(MODELDIR,'saved_model'), serving_input_receiver_fn)
