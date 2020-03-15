"""Train the model"""

import argparse
import os

import tensorflow as tf

# from model.input_fn import train_input_fn
# from model.input_fn import test_input_fn
from model.model_triplet_net import model_fn
# from model.utils import Params


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='C:\work\\tensorflow-triplet-loss\experiments\\batch_all',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='C:\work\\tensorflow-triplet-loss\data\\',
                    help="Directory containing the dataset")

def parse_exmp(serialized_example):
    feats = tf.parse_single_example(
        serialized_example,
        features={
            "text": tf.FixedLenFeature([FLAGS.sentence_max_len], tf.int64),
            # {"text": text, "label": label, "author": author, "categories": categories}
             "labels": tf.FixedLenFeature([12], tf.int64),
            "tags": tf.FixedLenFeature([12*10], tf.int64)
        })

    labels = feats.pop('label')

    return feats, labels


def train_input_fn(filenames, shuffle_buffer_size,shuffle=True,repeat=0):

    # Load txt file, one example per line
    files = tf.data.Dataset.list_files(filenames)  # A dataset of all files matching a pattern.
    dataset = files.apply(
        tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=FLAGS.num_parallel_readers))
    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(shuffle_buffer_size).repeat(FLAGS.train_epoch)
    dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=parse_exmp, batch_size=FLAGS.batch_size,
                                                          num_parallel_calls=2))
    return dataset


if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    # Load the parameters from json file
    args = parser.parse_args()
    # json_path = os.path.join(args.model_dir, 'params.json')
    # assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    # params = Params(json_path)

    # Define the model
    tf.logging.info("Creating the model...")
    config = tf.estimator.RunConfig(tf_random_seed=230,
                                    model_dir=args.model_dir,
                                    save_summary_steps=params.save_summary_steps)
    estimator = tf.estimator.Estimator(model_fn, params=params, config=config)

    # Train the model
    tf.logging.info("Starting training for {} epoch(s).".format(params.num_epochs))
    estimator.train(lambda: train_input_fn(args.data_dir))

    # Evaluate the model on the test set
    tf.logging.info("Evaluation on test set.")
    res = estimator.evaluate(lambda: test_input_fn(args.data_dir))
    for key in res:
        print("{}: {}".format(key, res[key]))