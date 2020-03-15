"""Train the model"""

import argparse
import os

import tensorflow as tf

# from model.input_fn import train_input_fn
# from model.input_fn import test_input_fn
from model.model_triplet_net import model_fn
from model.utils import Params


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='C:\work\\tensorflow-triplet-loss\experiments\\batch_all',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='C:\work\\tensorflow-triplet-loss\data\\',
                    help="Directory containing the dataset")

def train_input_fn(data_dir):
    """Train input function for the MNIST dataset.

    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    dataset = mnist_dataset.train(data_dir)
    dataset = dataset.shuffle(train_size)  # whole dataset into the buffer
    dataset = dataset.repeat(num_epochs)  # repeat for multiple epochs
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)  # make sure you always have one batch ready to serve
    return dataset

def test_input_fn(data_dir):
    """Test input function for the MNIST dataset.

    Args:
        data_dir: (string) path to the data directory
        params: (Params) contains hyperparameters of the model (ex: `params.num_epochs`)
    """
    dataset = mnist_dataset.test(data_dir)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)  # make sure you always have one batch ready to serve
    return dataset


if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

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
