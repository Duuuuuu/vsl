import pickle
import argparse
import logging

from sklearn.model_selection import train_test_split
from collections import Counter


def get_args():
    parser = argparse.ArgumentParser(
        description='Data Preprocessing for Named Entity Recognition')
    parser.add_argument('--train', type=str, default=None,
                        help='train data path')
    parser.add_argument('--dev', type=str, default=None,
                        help='dev data path')
    parser.add_argument('--test', type=str, default=None,
                        help='test data path')
    parser.add_argument('--ratio', type=float, default=1.,
                        help='ratio of labeled data')
    parser.add_argument('--crf', type=bool, default=False,
                        help='determine if the data is processed for vsl-crf')
    args = parser.parse_args()
    return args


def process(word):
    word = "".join(c if not c.isdigit() else '0' for c in word)
    return word


def process_file(data_file):
    logging.info("loading data from " + data_file + " ...")
    sents = []
    tags = []
    sent = []
    tag = []
    data_path = './data/'+data_file
    with open(data_path, 'r', encoding='utf-8') as df:
        for line in df.readlines():
            if line[0:10] == '-DOCSTART-':
                continue
            if line.strip():
                word = line.strip().split(" ")[0]
                t = line.strip().split(" ")[-1]
                sent.append(process(word))
                tag.append(t)
            else:
                if sent and tag:
                    sents.append(sent)
                    tags.append(tag)
                sent = []
                tag = []
    return sents, tags

def process_file_crf(data_file):
    logging.info("loading data from " + data_file + " ...")
    sents = []
    tags = []
    sent = []
    tag = []
    data_path = './data/'+data_file
    with open(data_path, 'r', encoding='utf-8') as df:
        for line in df.readlines():
            if line[0:10] == '-DOCSTART-':
                continue
            if line.strip():
                word = line.strip().split(" ")[0]
                t = line.strip().split(" ")[-1]
                sent.append(process(word))
                tag.append(t)
            else:
                if sent and tag:
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
        with open("ner/ner_tagfile_crf".format(args.ratio), "w+", encoding='utf-8') as fp:
            fp.write('\n'.join(sorted(tag_counter.keys())))
                
    
    else:
        train = process_file(args.train)
        dev = process_file(args.dev)
        test = process_file(args.test)
        tag_counter = Counter(sum(train[1], []) +
                          sum(dev[1], []) + sum(test[1], []))
        with open("ner/ner_tagfile".format(args.ratio), "w+", encoding='utf-8') as fp:
            fp.write('\n'.join(sorted(tag_counter.keys())))

    if args.ratio < 1:
        n_unlabel = len(train[0]) // 2
        X_train, X_test, y_train, y_test = \
            train_test_split(train[0], train[1], test_size=args.ratio)
        other = [X_train, y_train]
        train = [X_test, y_test]

        X_train, X_test, y_train, y_test = \
            train_test_split(other[0], other[1], test_size=n_unlabel)

        unlabel_data = X_test
        logging.info("#unlabeled data: {}".format(len(X_test)))
        
        unlabel_filename = f"ner/ner{args.ratio}_unlabel_{'crf' if args.crf else''}.data"
        with open(unlabel_filename,
                  "w+", encoding='utf-8') as fp:
            fp.write(
                "\n".join([" ".join([w for w in sent])
                          for sent in unlabel_data]))
        logging.info(
            "unlabeled data saved to {}".format(
                unlabel_filename))

    logging.info("#train data: {}".format(len(train[0])))
    logging.info("#dev data: {}".format(len(dev[0])))
    logging.info("#test data: {}".format(len(test[0])))

    filename = f"ner/ner{args.ratio}_{'crf' if args.crf else ''}.data"

    pickle.dump(
        [train, dev, test], open(filename, "wb+"),
        protocol=-1)