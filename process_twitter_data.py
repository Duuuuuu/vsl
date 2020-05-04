import pickle
import argparse
import logging

from sklearn.model_selection import train_test_split


def get_args():
    parser = argparse.ArgumentParser(
        description='Data Preprocessing for Twitter')
    parser.add_argument('--train', type=str, default=None,
                        help='train data path')
    parser.add_argument('--dev', type=str, default=None,
                        help='dev data path')
    parser.add_argument('--test', type=str, default=None,
                        help='test data path')
    parser.add_argument('--ratio', type=float, default=1.0,
                        help='training data ratio')
    parser.add_argument('--crf', type=bool, default=False,
                        help='determine if the data is processed for vsl-crf')
    args = parser.parse_args()
    return args


def process_file(data_file):
    logging.info("loading data from " + data_file + " ...")
    sents = []
    tags = []
    data_path = './data/'+data_file
    with open(data_path, 'r', encoding='utf-8') as df:
        for line in df.readlines():
            if line.strip():
                index = line.find('|||')
                if index == -1:
                    raise ValueError('Format Error')
                sent = line[: index - 1]
                tag = line[index + 4: -1]
                sents.append(sent.split(' '))
                tags.append(tag.split(' '))
    return sents, tags

def process_file_crf(data_file):
    logging.info("loading data from " + data_file + " ...")
    sents = []
    tags = []
    data_path = './data/'+data_file
    with open(data_path, 'r', encoding='utf-8') as df:
        for line in df.readlines():
            if line.strip():
                index = line.find('|||')
                if index == -1:
                    raise ValueError('Format Error')
                sent = '<start> '+line[: index - 1]+' <end>'
                tag = '<start> '+line[index + 4: -1]+' <end>'
                sents.append(sent.split(' '))
                tags.append(tag.split(' '))
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
        tag_set = set(sum([sum(d[1], []) for d in [train, dev, test]],
                  []))
        with open("twitter/twitter_tagfile_crf", "w+", encoding='utf-8') as fp:
            fp.write('\n'.join(sorted(list(tag_set))))
                
    
    else:
        train = process_file(args.train)
        dev = process_file(args.dev)
        test = process_file(args.test)
        tag_set = set(sum([sum(d[1], []) for d in [train, dev, test]],
                  []))
        with open("twitter/twitter_tagfile", "w+", encoding='utf-8') as fp:
            fp.write('\n'.join(sorted(list(tag_set))))


    if args.ratio != 1:
        train_x, test_x, train_y, test_y = \
            train_test_split(train[0], train[1], test_size=args.ratio)
        train = [test_x, test_y]
        assert len(train_x) == len(train_y)

    logging.info("#train: {}".format(len(train[0])))
    logging.info("#dev: {}".format(len(dev[0])))
    logging.info("#test: {}".format(len(test[0])))

    filename = f"data/twitter{args.ratio}_{'crf' if args.crf else ''}.data"

    pickle.dump(
        [train, dev, test],
        open(filename, "wb+"), protocol=-1)
