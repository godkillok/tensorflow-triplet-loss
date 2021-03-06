"""Build vocabularies of words and labels from datasets"""
import argparse
from collections import Counter
import json
import os
import re
from common_tool import per_line

parser = argparse.ArgumentParser()
parser.add_argument('--min_count_word', default=20, help="Minimum count for words in the dataset", type=int)
parser.add_argument('--data_dir', default='/data/tanggp/tmp/Starspace/python/test/',
                    help="Directory containing the dataset")

# Hyper parameters for the vocab
NUM_OOV_BUCKETS = 1  # number of buckets (= number of ids) for unknown words
PAD_WORD = '0'
label_class = []


def save_vocab_to_txt_file(vocab, txt_path):
    """
    Writes one token per line, 0-based line id corresponds to the id of the token.
    Args:
        vocab: (iterable object) yields token
        txt_path: (stirng) path to vocab file
    """
    with open(txt_path, "w", encoding="utf8") as f:
        f.write("\n".join(token for token in vocab))


def save_label_to_txt_file(labels, txt_path):
    """
    Writes one token per line, 0-based line id corresponds to the id of the token.
    Args:
        vocab: (iterable object) yields token
        txt_path: (stirng) path to vocab file
    """
    with open(txt_path, "w", encoding="utf8") as f:
        for vo in labels:
            f.write("{}\x01\t{}\n".format(vo[0], vo[1]))


def save_dict_to_json(d, json_path):
    """Saves dict to json file
    Args:
        d: (dict)
        json_path: (string) path to json file
    """
    with open(json_path, 'w', encoding="utf8") as f:
        d = {k: v for k, v in d.items()}
        json.dump(d, f, indent=4)


def update_label(txt_path, labels):
    """Update word and tag vocabulary from dataset
    Args:
        txt_path: (string) path to file, one sentence per line
        vocab: (dict or Counter) with update method
    Returns:
        dataset_size: (int) number of elements in the dataset
    """
    with open(txt_path, "r", encoding="utf8") as f:
        for i, line in enumerate(f):
            li = line.strip()
            li = li.split("__label__")
            for l in li[1:]:
                punctuation = r"""0123456789"""
                text = re.sub(r'[{}]+'.format(punctuation), ' ', str(l))
                text = ' '.join(text.split())
                text = text.lower()
                text = ' '.join(text.split("\x01_"))
                text = ' '.join(text.split())
                labels.append(text.strip())


if __name__ == '__main__':
    args = parser.parse_args()

    # Build word vocab with train and test datasets
    print("Building labels...")
    labels = []
    update_label(os.path.join(args.data_dir, 'tag_space'), labels)
    # update_label(os.path.join(args.data_dir, 'txt_golden'), labels)
    # update_label(os.path.join(args.data_dir, 'txt_valid'), labels)
    labels_sort = sorted(Counter(labels).items(), key=lambda x: x[1], reverse=True)
    print('labels num {}'.format(len(labels_sort)))
    save_label_to_txt_file(labels_sort, os.path.join(args.data_dir, 'textcnn_label_sort'))
    print("- done.")
    os.system("cat {}".format(os.path.join(args.data_dir, 'textcnn_label_sort')))
    print("==" * 8)
    os.system("tail {}".format(os.path.join(args.data_dir, 'textcnn_label_sort')))
