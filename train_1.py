"""Train the model"""

import argparse
import os

import tensorflow as tf

# from model.input_fn import train_input_fn
# from model.input_fn import test_input_fn
from model.model_triplet_net import model_fn
from model.utils import Params

os.environ['CUDA_VISIBLE_DEVICES'] ="8"
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='/data/tanggp/tmp/tensorflow-triplet-loss/experiments/batch_all_2',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='/data/tanggp/tmp/Starspace/python/test/*.tfrecords',
                    help="Directory containing the dataset")
num_parallel_readers=4
batch_size=128
num_epochs=20
def parse_exmp(serialized_example):
    feats = tf.parse_single_example(
        serialized_example,
        features={
            "text": tf.FixedLenFeature([60], tf.int64),
            # {"text": text, "label": label, "author": author, "categories": categories}
             "labels": tf.FixedLenFeature([12], tf.int64),
            "tags": tf.FixedLenFeature([120], tf.int64)
        })

    #labels = feats.pop('label')

    return feats


def train_input_fn(filenames, shuffle_buffer_size=5,shuffle=True,repeat=0):

    # Load txt file, one example per line
    files = tf.data.Dataset.list_files(filenames)  # A dataset of all files matching a pattern.
    print("files {}".format(files))
    dataset = files.apply(
        tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=num_parallel_readers))
    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(shuffle_buffer_size).repeat(num_epochs)
    dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=parse_exmp, batch_size=batch_size))
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
                                    save_summary_steps=5
                                    )
    session_config = tf.ConfigProto(log_device_placement=True)
    session_config.gpu_options.allow_growth = True
    run_config = tf.estimator.RunConfig().replace(session_config=session_config)

    # config.gpu_options.allow_growth = True
    #json_path = os.path.join(args.model_dir, 'params.json')
    # assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    # params = Params(json_path)
    params=None
    estimator = tf.estimator.Estimator(model_fn, params=params, config=config)

    # Train the model
    tf.logging.info("Starting training for {} epoch(s).".format(num_epochs))
    estimator.train(lambda: train_input_fn(args.data_dir))

    # Evaluate the model on the test set
    tf.logging.info("Evaluation on test set.")
    res = estimator.evaluate(lambda: train_input_fn(args.data_dir))
    for key in res:
        print("{}: {}".format(key, res[key]))

    # Evaluate the model on the test set
    # tf.logging.info("Predition on pred set.")
    # res = estimator.predict(lambda: train_input_fn(args.data_dir))
    # print(res)
    # for key in res:
    #     print(key)
    # print(res)
