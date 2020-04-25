from collections import Counter
from functools import reduce
import yaml

import codecs
import pickle

import numpy as np

import torch
import torch.nn as nn
import torch.nn.init

CONFIG_PATH = "./config.yml"

def load_config(filename):
    with open(filename, 'r', encoding='utf-8') as fin:
        config = yaml.load(fin, Loader=yaml.SafeLoader)
    return config

config = load_config(CONFIG_PATH)

def load_words_tags(filename, data_type='train'):
    with open(filename, 'rb') as fin:
        data = pickle.load(fin)

    if data_type == 'train':
        return data[0][0], data[0][1]
    elif data_type == 'dev':
        return data[1][0], data[1][1]
    elif data_type == 'test':
        return data[2][0], data[2][1]
    else:
        raise ValueError("data_type must be one of [train|dev|test]")


def read_words_tags(file, tag_ind, caseless=True):
    """
    Reads raw data in the CoNLL 2003 format and returns word and tag sequences.

    :param file: file with raw data in the CoNLL 2003 format
    :param tag_ind: column index of tag
    :param caseless: lowercase words?
    :return: word, tag sequences
    """
    with codecs.open(file, 'r', 'utf-8') as f:
        lines = f.readlines()
    words = []
    tags = []
    temp_w = []
    temp_t = []
    for line in lines:
        if not (line.isspace() or (len(line) > 10 and line[0:10] == '-DOCSTART-')):
            feats = line.rstrip('\n').split()
            temp_w.append(feats[0].lower() if caseless else feats[0])
            temp_t.append(feats[tag_ind])
        elif len(temp_w) > 0:
            assert len(temp_w) == len(temp_t)
            words.append(temp_w)
            tags.append(temp_t)
            temp_w = []
            temp_t = []
    # last sentence
    if len(temp_w) > 0:
        assert len(temp_w) == len(temp_t)
        words.append(temp_w)
        tags.append(temp_t)

    # Sanity check
    assert len(words) == len(tags)

    return words, tags


def create_maps(words, tags, min_word_freq=5, min_char_freq=1):
    """
    Creates word, char, tag maps.

    :param words: word sequences
    :param tags: tag sequences
    :param min_word_freq: words that occur fewer times than this threshold are binned as <unk>s
    :param min_char_freq: characters that occur fewer times than this threshold are binned as <unk>s
    :return: word, char, tag maps
    """
    word_freq = Counter()
    char_freq = Counter()
    tag_map = set()
    for w, t in zip(words, tags):
        word_freq.update(w)
        char_freq.update(list(reduce(lambda x, y: list(x) + [' '] + list(y), w)))
        tag_map.update(t)

    word_map = {k: v + 1 for v, k in enumerate([w for w in word_freq.keys() if word_freq[w] > min_word_freq])}
    char_map = {k: v + 1 for v, k in enumerate([c for c in char_freq.keys() if char_freq[c] > min_char_freq])}
    tag_map = {k: v + 1 for v, k in enumerate(tag_map)}

    word_map['<pad>'] = 0
    word_map['<end>'] = len(word_map)
    word_map['<unk>'] = len(word_map)
    char_map['<pad>'] = 0
    char_map['<end>'] = len(char_map)
    char_map['<unk>'] = len(char_map)
    char_map[' '] = 0
    tag_map['<pad>'] = 0
    tag_map['<start>'] = len(tag_map)
    tag_map['<end>'] = len(tag_map)

    return word_map, char_map, tag_map

def prepare_char_maps(words, padded_wmaps, char_map):
    assert len(words) == len(padded_wmaps)

    lens = [[len(word) for word in sentence] + [1] for sentence in words]
    maxlen_s = max([len(li) for li in lens])
    maxlen_w = max([max(li_s) for li_s in lens])

    assert maxlen_s == padded_wmaps.size(1)

    N = len(words)
    output = np.empty((N, maxlen_s, maxlen_w), dtype='int')
    output.fill(char_map['<pad>'])
    for i in range(N):
        for j in range(len(words[i])):
            output[i, j, :lens[i][j]] = [char_map.get(c, char_map['<unk>']) for c in words[i][j]]
        output[i, len(words[i]), 0] = char_map['<end>']

    output_lens = torch.LongTensor(N, maxlen_s)
    output_lens.fill_(0)
    for i in range(N):
        for j in range(len(words[i])):
            output_lens[i, j] = lens[i][j]
        output_lens[i, len(words[i])] = 1

    output = torch.from_numpy(output)

    assert padded_wmaps.size(0) == output.size(0)
    assert padded_wmaps.size(1) == output.size(1)

    return output, output_lens

def create_input_tensors(words, tags, word_map, char_map, tag_map):
    """
    Creates input tensors that will be used to create a PyTorch Dataset.

    :param words: word sequences
    :param tags: tag sequences
    :param word_map: word map
    :param char_map: character map
    :param tag_map: tag map
    :return: padded encoded words, padded encoded forward chars, padded encoded backward chars,
            padded forward character markers, padded backward character markers, padded encoded tags,
            word sequence lengths, char sequence lengths
    """
    assert len(words) == len(tags)
    # Encode sentences into word maps with <end> at the end
    # [['dunston', 'checks', 'in', '<end>']] -> [[4670, 4670, 185, 4669]]
    wmaps = list(map(lambda s: list(map(lambda w: word_map.get(w, word_map['<unk>']), s)) + [word_map['<end>']], words))

    # Forward and backward character streams
    # [['d', 'u', 'n', 's', 't', 'o', 'n', ' ', 'c', 'h', 'e', 'c', 'k', 's', ' ', 'i', 'n', ' ']]
    chars_f = list(map(lambda s: list(reduce(lambda x, y: list(x) + [' '] + list(y), s)) + [' '], words))
    # [['n', 'i', ' ', 's', 'k', 'c', 'e', 'h', 'c', ' ', 'n', 'o', 't', 's', 'n', 'u', 'd', ' ']]
    chars_b = list(
        map(lambda s: list(reversed([' '] + list(reduce(lambda x, y: list(x) + [' '] + list(y), s)))), words))

    # Encode streams into forward and backward character maps with <end> at the end
    # [[29, 2, 12, 8, 7, 14, 12, 3, 6, 18, 1, 6, 21, 8, 3, 17, 12, 3, 60]]
    cmaps_f = list(
        map(lambda s: list(map(lambda c: char_map.get(c, char_map['<unk>']), s)) + [char_map['<end>']], chars_f))
    # [[12, 17, 3, 8, 21, 6, 1, 18, 6, 3, 12, 14, 7, 8, 12, 2, 29, 3, 60]]
    cmaps_b = list(
        map(lambda s: list(map(lambda c: char_map.get(c, char_map['<unk>']), s)) + [char_map['<end>']], chars_b))
    
    # Positions of spaces and <end> character
    # Words are predicted or encoded at these places in the language and tagging models respectively
    # [[7, 14, 17, 18]] are points after '...dunston', '...checks', '...in', '...<end>' respectively
    cmarkers_f = list(map(lambda s: [ind for ind in range(len(s)) if s[ind] == char_map[' ']] + [len(s) - 1], cmaps_f))
    # Reverse the markers for the backward stream before adding <end>, so the words of the f and b markers coincide
    # i.e., [[17, 9, 2, 18]] are points after '...notsnud', '...skcehc', '...ni', '...<end>' respectively
    cmarkers_b = list(
        map(lambda s: list(reversed([ind for ind in range(len(s)) if s[ind] == char_map[' ']])) + [len(s) - 1],
            cmaps_b))

    # Encode tags into tag maps with <end> at the end
    tmaps = list(map(lambda s: list(map(lambda t: tag_map[t], s)) + [tag_map['<end>']], tags))
    # Since we're using CRF scores of size (prev_tags, cur_tags), find indices of target sequence in the unrolled scores
    # This will be row_index (i.e. prev_tag) * n_columns (i.e. tagset_size) + column_index (i.e. cur_tag)
    tmaps = list(map(lambda s: [tag_map['<start>'] * len(tag_map) + s[0]] + [s[i - 1] * len(tag_map) + s[i] for i in
                                                                            range(1, len(s))], tmaps))
    # Note - the actual tag indices can be recovered with tmaps % len(tag_map)

    # Pad, because need fixed length to be passed around by DataLoaders and other layers
    word_pad_len = max(list(map(lambda s: len(s), wmaps)))
    char_pad_len = max(list(map(lambda s: len(s), cmaps_f)))

    # Sanity check
    assert word_pad_len == max(list(map(lambda s: len(s), tmaps)))

    padded_wmaps = []
    padded_cmaps_f = []
    padded_cmaps_b = []
    padded_cmarkers_f = []
    padded_cmarkers_b = []
    padded_tmaps = []
    wmap_lengths = []
    cmap_lengths = []

    line_no = 0
    for w, cf, cb, cmf, cmb, t in zip(wmaps, cmaps_f, cmaps_b, cmarkers_f, cmarkers_b, tmaps):
        # Sanity  checks
        assert len(w) == len(cmf) == len(cmb) == len(t)
        assert len(cmaps_f) == len(cmaps_b)

        # Pad
        # A note -  it doesn't really matter what we pad with, as long as it's a valid index
        # i.e., we'll extract output at those pad points (to extract equal lengths), but never use them

        padded_wmaps.append(w + [word_map['<pad>']] * (word_pad_len - len(w)))
        padded_cmaps_f.append(cf + [char_map['<pad>']] * (char_pad_len - len(cf)))
        padded_cmaps_b.append(cb + [char_map['<pad>']] * (char_pad_len - len(cb)))

        # 0 is always a valid index to pad markers with (-1 is too but torch.gather has some issues with it)
        padded_cmarkers_f.append(cmf + [0] * (word_pad_len - len(w)))
        padded_cmarkers_b.append(cmb + [0] * (word_pad_len - len(w)))

        padded_tmaps.append(t + [tag_map['<pad>']] * (word_pad_len - len(t)))

        wmap_lengths.append(len(w))
        cmap_lengths.append(len(cf))

        # Sanity check
        assert len(padded_wmaps[-1]) == len(padded_tmaps[-1]) == len(padded_cmarkers_f[-1]) == len(
            padded_cmarkers_b[-1]) == word_pad_len
        assert len(padded_cmaps_f[-1]) == len(padded_cmaps_b[-1]) == char_pad_len

        line_no += 1

    padded_wmaps = torch.LongTensor(padded_wmaps)
    padded_cmaps_f = torch.LongTensor(padded_cmaps_f)
    padded_cmaps_b = torch.LongTensor(padded_cmaps_b)
    padded_cmarkers_f = torch.LongTensor(padded_cmarkers_f)
    padded_cmarkers_b = torch.LongTensor(padded_cmarkers_b)
    padded_tmaps = torch.LongTensor(padded_tmaps)
    wmap_lengths = torch.LongTensor(wmap_lengths)
    cmap_lengths = torch.LongTensor(cmap_lengths)

    padded_cmaps, cmap_w_lengths = prepare_char_maps(words, padded_wmaps, char_map)

    return padded_wmaps, padded_cmaps, padded_cmaps_f, padded_cmaps_b, padded_cmarkers_f, padded_cmarkers_b, padded_tmaps, \
            wmap_lengths, cmap_lengths, cmap_w_lengths


def init_embedding(input_embedding):
    """
    refer: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Sequence-Labeling

    Initialize embedding tensor with values from the uniform distribution.
    :param input_embedding: embedding tensor
    :return:
    """
    bias = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform_(input_embedding, -bias, bias)


def load_embeddings(emb_file, word_map, expand_vocab=True):
    """
    refer: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Sequence-Labeling

    Load pre-trained embeddings for words in the word map.
    :param emb_file: file with pre-trained embeddings (in the GloVe format)
    :param word_map: word map
    :param expand_vocab: expand vocabulary of word map to vocabulary of pre-trained embeddings?
    :return: embeddings for words in word map, (possibly expanded) word map,
            number of words in word map that are in-corpus (subject to word frequency threshold)
    """
    with open(emb_file, 'r') as f:
        emb_len = len(f.readline().split()) - 1

    print("Embedding length is %d." % emb_len)

    # Create tensor to hold embeddings for words that are in-corpus
    ic_embs = torch.FloatTensor(len(word_map), emb_len)
    init_embedding(ic_embs)

    if expand_vocab:
        print("You have elected to include embeddings that are out-of-corpus.")
        ooc_words = []
        ooc_embs = []
    else:
        print("You have elected NOT to include embeddings that are out-of-corpus.")

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'):
        line = line.split('\t')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1].split(' '))))

        if not expand_vocab and emb_word not in word_map:
            continue

        # If word is in train_vocab, store at the correct index (as in the word_map)
        if emb_word in word_map:
            ic_embs[word_map[emb_word]] = torch.FloatTensor(embedding)

        # If word is in dev or test vocab, store it and its embedding into lists
        elif expand_vocab:
            ooc_words.append(emb_word)
            ooc_embs.append(embedding)

    lm_vocab_size = len(word_map)  # keep track of lang. model's output vocab size (no out-of-corpus words)

    if expand_vocab:
        print("'word_map' is being updated accordingly.")
        for word in ooc_words:
            word_map[word] = len(word_map)
        ooc_embs = torch.FloatTensor(np.asarray(ooc_embs))
        embeddings = torch.cat([ic_embs, ooc_embs], 0)

    else:
        embeddings = ic_embs

    # Sanity check
    assert embeddings.size(0) == len(word_map)

    print("\nDone.\n Embedding vocabulary: %d\n Language Model vocabulary: %d.\n" % (len(word_map), lm_vocab_size))

    return embeddings, word_map, lm_vocab_size


def clip_gradient(optimizer, grad_clip):
    """
    refer: https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Sequence-Labeling

    Clip gradients computed during backpropagation to prevent gradient explosion.
    :param optimizer: optimized with the gradients to be clipped
    :param grad_clip: gradient clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(epoch, model, optimizer, metric_val, word_map, char_map, tag_map, lm_vocab_size, is_best):
    """
    Save model checkpoint.

    :param epoch: epoch number
    :param model: model
    :param optimizer: optimized
    :param metric_val: validation F1 score
    :param word_map: word map
    :param char_map: char map
    :param tag_map: tag map
    :param lm_vocab_size: number of words in-corpus, i.e. size of output vocabulary of linear model
    :param is_best: is this checkpoint the best so far?
    :return:
    """
    state = {
        'epoch': epoch,
        'metric': metric_val,
        'model': model,
        'optimizer': optimizer,
        'word_map': word_map,
        'tag_map': tag_map,
        'char_map': char_map,
        'lm_vocab_size': lm_vocab_size
    }
    filename = config['ckpt_path']
    torch.save(state, filename)
    # If checkpoint is the best so far, create a copy to avoid being overwritten by a subsequent worse checkpoint
    if is_best:
        torch.save(state, config['best_ckpt_path'])


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, new_lr):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rates must be decayed
    :param new_lr: new learning rate
    """

    # print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    # print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def log_sum_exp(tensor, dim):
    """
    Calculates the log-sum-exponent of a tensor's dimension in a numerically stable way.

    :param tensor: tensor
    :param dim: dimension to calculate log-sum-exp of
    :return: log-sum-exp
    """
    m, _ = torch.max(tensor, dim)
    m_expanded = m.unsqueeze(dim).expand_as(tensor)
    return m + torch.log(torch.sum(torch.exp(tensor - m_expanded), dim))
