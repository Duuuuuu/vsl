import pickle
import argparse
import logging
from collections import Counter
from sklearn.model_selection import train_test_split


def get_args():
    parser = argparse.ArgumentParser(
        description='Data Preprocessing for Universal Dependencies')
    parser.add_argument('--train', type=str, default=None,
                        help='train data path')
    parser.add_argument('--dev', type=str, default=None,
                        help='dev data path')
    parser.add_argument('--test', type=str, default=None,
                        help='test data path')
    parser.add_argument('--output', type=str, default=None,
                        help='output file name')
    parser.add_argument('--ratio', type=float, default=1.0,
                        help='labeled data ratio')
    parser.add_argument('--crf', type=bool, default=False,
                        help='determine if the data is processed for vsl-crf')
    args = parser.parse_args()
    return args


def process_file(data_file):
    logging.info("loading data from {} ...".format(data_file))
    sents = []
    tags = []
    sent = []
    tag = []
    data_path = './data/'+data_file
    with open(data_path, 'r', encoding="utf-8") as f:
        for line in f:
            if line[0] == "#":
                continue
            if line.strip():
                token = line.strip("\n").split("\t")
                word = token[1]
                t = token[3]
                sent.append(word)
                tag.append(t)
            else:
                sents.append(sent)
                tags.append(tag)
                sent = []
                tag = []
    return sents, tags

def process_file_crf(data_file):
    logging.info("loading data from {} ...".format(data_file))
    sents = []
    tags = []
    sent = []
    tag = []
    data_path = './data/'+data_file
    with open(data_path, 'r', encoding="utf-8") as f:
        for line in f:
            if line[0] == "#":
                continue
            if line.strip():
                token = line.strip("\n").split("\t")
                word = token[1]
                t = token[3]
                sent.append(word)
                tag.append(t)
            else:
                sent = ['<start>']+sent+['<end>']
                tag = ['<start>']+tag+['<end>']
                sents.append(sent)
                tags.append(tag)
                sent = []
                tag = []
    return sents, tags


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%m-%d %H:%M')
    args = get_args()

    if args.crf:
        train = process_file_crf(args.train)
        dev = process_file_crf(args.dev)
        test = process_file_crf(args.test)
        tag_counter = Counter(sum(train[1], []) +
                          sum(dev[1], []) + sum(test[1], []))
        with open("ud/ud_tagfile_crf".format(args.ratio), "w+", encoding='utf-8') as fp:
            fp.write('\n'.join(sorted(tag_counter.keys())))
                
    
    else:
        train = process_file(args.train)
        dev = process_file(args.dev)
        test = process_file(args.test)
        tag_counter = Counter(sum(train[1], []) +
                          sum(dev[1], []) + sum(test[1], []))
        with open("ud/ud_tagfile".format(args.ratio), "w+", encoding='utf-8') as fp:
            fp.write('\n'.join(sorted(tag_counter.keys())))


    if args.ratio != 1:
        train_x, test_x, train_y, test_y = \
            train_test_split(train[0], train[1], test_size=args.ratio)
        train = [test_x, test_y]
        assert len(train_x) == len(train_y)

    logging.info("#train: {}".format(len(train[0])))
    logging.info("#dev: {}".format(len(dev[0])))
    logging.info("#test: {}".format(len(test[0])))

    filename = f"data/ud{args.ratio}_{'crf' if args.crf else ''}.data"

    pickle.dump(
        [train, dev, test],
        open(filename, "wb+"), protocol=-1)
