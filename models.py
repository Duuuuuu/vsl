import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Highway(nn.Module):
    """
    Highway Network.
    """

    def __init__(self, size, num_layers=1, dropout=0.5):
        """
        :param size: size of linear layer (matches input size)
        :param num_layers: number of transform and gate layers
        :param dropout: dropout
        """
        super(Highway, self).__init__()
        self.size = size
        self.num_layers = num_layers
        self.transform = nn.ModuleList()  # list of transform layers
        self.gate = nn.ModuleList()  # list of gate layers
        self.dropout = nn.Dropout(p=dropout)

        for i in range(num_layers):
            transform = nn.Linear(size, size)
            gate = nn.Linear(size, size)
            self.transform.append(transform)
            self.gate.append(gate)

    def forward(self, x):
        """
        Forward propagation.

        :param x: input tensor
        :return: output tensor, with same dimensions as input tensor
        """
        transformed = nn.functional.relu(self.transform[0](x))  # transform input
        g = nn.functional.sigmoid(self.gate[0](x))  # calculate how much of the transformed input to keep

        out = g * transformed + (1 - g) * x  # combine input and transformed input in this ratio

        # If there are additional layers
        for i in range(1, self.num_layers):
            out = self.dropout(out)
            transformed = nn.functional.relu(self.transform[i](out))
            g = nn.functional.sigmoid(self.gate[i](out))

            out = g * transformed + (1 - g) * out

        return out


class CRF(nn.Module):
    """
    Conditional Random Field.
    """

    def __init__(self, hidden_dim, tagset_size, tag_map):
        """
        :param hidden_dim: size of word RNN/BLSTM's output
        :param tagset_size: number of tags
        """
        super(CRF, self).__init__()
        self.tagset_size = tagset_size
        self.emission = nn.Linear(hidden_dim, self.tagset_size)
        self.init_linear(self.emission)

        self.transition = nn.Parameter(torch.Tensor(self.tagset_size, self.tagset_size))
        self.transition.data.zero_()
        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transition.data[tag_map['<start>'], :] = -10000
        self.transition.data[:, tag_map['<end>']] = -10000

    def forward(self, feats):
        """
        Forward propagation.

        :param feats: output of word RNN/BLSTM, a tensor of dimensions (batch_size, timesteps, hidden_dim)
        :return: CRF scores, a tensor of dimensions (batch_size, timesteps, tagset_size, tagset_size)
        """
        self.batch_size = feats.size(0)
        self.timesteps = feats.size(1)

        emission_scores = self.emission(feats)  # (batch_size, timesteps, tagset_size)
        emission_scores = emission_scores.unsqueeze(2).expand(self.batch_size, self.timesteps, self.tagset_size,
                                                              self.tagset_size)  # (batch_size, timesteps, tagset_size, tagset_size)

        crf_scores = emission_scores + self.transition.unsqueeze(0).unsqueeze(
            0)  # (batch_size, timesteps, tagset_size, tagset_size)
        return crf_scores

    def init_linear(self, input_linear):
        """
        Initialize linear transformation
        """
        bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
        nn.init.uniform_(input_linear.weight, -bias, bias)
        if input_linear.bias is not None:
            input_linear.bias.data.zero_()


class LM_LSTM_CRF(nn.Module):
    """
    The encompassing LM-LSTM-CRF model.
    """

    def __init__(self, tagset_size, tagset_map, charset_size, char_emb_dim, char_feat_dim, char_rnn_layers, vocab_size,
                 lm_vocab_size, word_emb_dim, word_rnn_dim, word_rnn_layers, dropout, highway_layers=1, char_type='lstm'):
        """
        :param tagset_size: number of tags
        :param charset_size: size of character vocabulary
        :param char_emb_dim: size of character embeddings
        :param char_feat_dim: size of character RNNs/LSTMs
        :param char_rnn_layers: number of layers in character RNNs/LSTMs
        :param vocab_size: input vocabulary size
        :param lm_vocab_size: vocabulary size of language models (in-corpus words subject to word frequency threshold)
        :param word_emb_dim: size of word embeddings
        :param word_rnn_dim: size of word RNN/BLSTM
        :param word_rnn_layers:  number of layers in word RNNs/LSTMs
        :param dropout: dropout
        :param highway_layers: number of transform and gate layers
        """

        super(LM_LSTM_CRF, self).__init__()

        self.tagset_size = tagset_size  # this is the size of the output vocab of the tagging model
        self.tagset_map = tagset_map

        self.charset_size = charset_size
        self.char_emb_dim = char_emb_dim
        self.char_feat_dim = char_feat_dim
        self.char_rnn_layers = char_rnn_layers

        self.wordset_size = vocab_size  # this is the size of the input vocab (embedding layer) of the tagging model
        self.lm_vocab_size = lm_vocab_size  # this is the size of the output vocab of the language model
        self.word_emb_dim = word_emb_dim
        self.word_rnn_dim = word_rnn_dim
        self.word_rnn_layers = word_rnn_layers

        self.char_type = char_type

        self.highway_layers = highway_layers

        self.dropout = nn.Dropout(p=dropout)

        self.char_embeds = nn.Embedding(self.charset_size, self.char_emb_dim)  # character embedding layer
        init_embedding(self.char_embeds.weight)

        self.word_embeds = nn.Embedding(self.wordset_size, self.word_emb_dim)  # word embedding layer
        init_embedding(self.word_embeds.weight)

        if self.char_type == 'lstm':
            self.forw_char_lstm = nn.LSTM(self.char_emb_dim, self.char_feat_dim, num_layers=self.char_rnn_layers,
                                        bidirectional=False, dropout=dropout)  # forward character LSTM
            self.back_char_lstm = nn.LSTM(self.char_emb_dim, self.char_feat_dim, num_layers=self.char_rnn_layers,
                                        bidirectional=False, dropout=dropout)  # backward character LSTM
            self.forw_lm_hw = Highway(self.char_feat_dim, num_layers=self.highway_layers,
                                    dropout=dropout)  # highway to transform forward char LSTM output for the forward language model
            self.back_lm_hw = Highway(self.char_feat_dim, num_layers=self.highway_layers,
                                    dropout=dropout)  # highway to transform backward char LSTM output for the backward language model
            self.subword_hw = Highway(2 * self.char_feat_dim, num_layers=self.highway_layers,
                                    dropout=dropout)  # highway to transform combined forward and backward char LSTM outputs for use in the word BLSTM

            self.forw_lm_out = nn.Linear(self.char_feat_dim,
                                        self.lm_vocab_size)  # linear layer to find vocabulary scores for the forward language model
            self.back_lm_out = nn.Linear(self.char_feat_dim,
                                        self.lm_vocab_size)  # linear layer to find vocabulary scores for the backward language model

            self.word_blstm = nn.LSTM(self.word_emb_dim + self.char_feat_dim * 2, self.word_rnn_dim,
                                    num_layers=self.word_rnn_layers, bidirectional=True, dropout=dropout)  # word BLSTM

        elif self.char_type == 'cnn':
            self.char_cnn3 = nn.Conv2d(in_channels=1, out_channels=self.char_feat_dim, kernel_size=(3, self.char_emb_dim), padding=(2,0))

            self.word_blstm = nn.LSTM(self.word_emb_dim + self.char_feat_dim, self.word_rnn_dim,
                                    num_layers=self.word_rnn_layers, bidirectional=True)  # word BLSTM
        else:
            self.word_blstm = nn.LSTM(self.word_emb_dim, self.word_rnn_dim,
                                    num_layers=self.word_rnn_layers, bidirectional=True)  # word BLSTM


        self.init_lstm(self.word_blstm)

        self.crf = CRF(self.word_rnn_dim * 2, self.tagset_size, self.tagset_map)  # conditional random field

    def init_word_embeddings(self, embeddings):
        """
        Initialize embeddings with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.word_embeds.weight = nn.Parameter(embeddings)

    def fine_tune_word_embeddings(self, fine_tune=False):
        """
        Fine-tune embedding layer? (Not fine-tuning only makes sense if using pre-trained embeddings).

        :param fine_tune: Fine-tune?
        """
        for p in self.word_embeds.parameters():
            p.requires_grad = fine_tune

    def forward(self, cmaps_f, cmaps_b, cmarkers_f, cmarkers_b, wmaps, cmaps, tmaps, wmap_lengths, cmap_lengths, cmap_w_lengths):
        """
        Forward propagation.

        :param cmaps_f: padded encoded forward character sequences, a tensor of dimensions (batch_size, char_pad_len)
        :param cmaps_b: padded encoded backward character sequences, a tensor of dimensions (batch_size, char_pad_len)
        :param cmarkers_f: padded forward character markers, a tensor of dimensions (batch_size, word_pad_len)
        :param cmarkers_b: padded backward character markers, a tensor of dimensions (batch_size, word_pad_len)
        :param wmaps: padded encoded word sequences, a tensor of dimensions (batch_size, word_pad_len)
        :param tmaps: padded tag sequences, a tensor of dimensions (batch_size, word_pad_len)
        :param wmap_lengths: word sequence lengths, a tensor of dimensions (batch_size)
        :param cmap_lengths: character sequence lengths, a tensor of dimensions (batch_size, word_pad_len)
        """
        self.batch_size = wmaps.size(0)
        self.word_pad_len = wmaps.size(1)

        if self.char_type == 'lstm':
            # Sort by decreasing true char. sequence length
            cmap_lengths, char_sort_ind = cmap_lengths.sort(dim=0, descending=True)
            cmaps_f = cmaps_f[char_sort_ind]
            cmaps_b = cmaps_b[char_sort_ind]
            cmarkers_f = cmarkers_f[char_sort_ind]
            cmarkers_b = cmarkers_b[char_sort_ind]
            wmaps = wmaps[char_sort_ind]
            tmaps = tmaps[char_sort_ind]
            wmap_lengths = wmap_lengths[char_sort_ind]

            # Embedding look-up for characters
            cf = self.char_embeds(cmaps_f)  # (batch_size, char_pad_len, char_emb_dim)
            cb = self.char_embeds(cmaps_b)

            # Dropout
            cf = self.dropout(cf)  # (batch_size, char_pad_len, char_emb_dim)
            cb = self.dropout(cb)

            # Pack padded sequence
            cf = pack_padded_sequence(cf, cmap_lengths.tolist(), batch_first=True)  # packed sequence of char_emb_dim, with real sequence lengths
            cb = pack_padded_sequence(cb, cmap_lengths.tolist(), batch_first=True)

            # LSTM
            cf, _ = self.forw_char_lstm(cf)  # packed sequence of char_feat_dim, with real sequence lengths
            cb, _ = self.back_char_lstm(cb)

            # Unpack packed sequence
            cf, _ = pad_packed_sequence(cf, batch_first=True)  # (batch_size, max_char_len_in_batch, char_feat_dim)
            cb, _ = pad_packed_sequence(cb, batch_first=True)

            # Sanity check
            assert cf.size(1) == max(cmap_lengths.tolist()) == list(cmap_lengths)[0]

            # Select RNN outputs only at marker points (spaces in the character sequence)
            cmarkers_f = cmarkers_f.unsqueeze(2).expand(self.batch_size, self.word_pad_len, self.char_feat_dim)
            cmarkers_b = cmarkers_b.unsqueeze(2).expand(self.batch_size, self.word_pad_len, self.char_feat_dim)
            cf_selected = torch.gather(cf, 1, cmarkers_f)  # (batch_size, word_pad_len, char_feat_dim)
            cb_selected = torch.gather(cb, 1, cmarkers_b)

            # Only for co-training, not useful for tagging after model is trained
            if self.training:
                lm_f = self.forw_lm_hw(self.dropout(cf_selected))  # (batch_size, word_pad_len, char_feat_dim)
                lm_b = self.back_lm_hw(self.dropout(cb_selected))
                lm_f_scores = self.forw_lm_out(self.dropout(lm_f))  # (batch_size, word_pad_len, lm_vocab_size)
                lm_b_scores = self.back_lm_out(self.dropout(lm_b))

            # Sort by decreasing true word sequence length
            wmap_lengths, word_sort_ind = wmap_lengths.sort(dim=0, descending=True)
            wmaps = wmaps[word_sort_ind]
            tmaps = tmaps[word_sort_ind]
            cf_selected = cf_selected[word_sort_ind]  # for language model
            cb_selected = cb_selected[word_sort_ind]
            if self.training:
                lm_f_scores = lm_f_scores[word_sort_ind]
                lm_b_scores = lm_b_scores[word_sort_ind]

            # Embedding look-up for words
            w = self.word_embeds(wmaps)  # (batch_size, word_pad_len, word_emb_dim)
            w = self.dropout(w)

            subword = self.subword_hw(self.dropout(
                torch.cat((cf_selected, cb_selected), dim=2)))  # (batch_size, word_pad_len, 2 * char_feat_dim)
            subword = self.dropout(subword)

            # Concatenate word embeddings and sub-word features
            w = torch.cat((w, subword), dim=2)  # (batch_size, word_pad_len, word_emb_dim + 2 * char_feat_dim)

            # Pack padded sequence
            w = pack_padded_sequence(w, list(wmap_lengths),
                                    batch_first=True)  # packed sequence of word_emb_dim + 2 * char_feat_dim, with real sequence lengths

            # LSTM
            w, _ = self.word_blstm(w)  # packed sequence of word_rnn_dim, with real sequence lengths

            # Unpack packed sequence
            w, _ = pad_packed_sequence(w, batch_first=True)  # (batch_size, max_word_len_in_batch, word_rnn_dim)
            w = self.dropout(w)

            crf_scores = self.crf(w)  # (batch_size, max_word_len_in_batch, tagset_size, tagset_size)

        elif self.char_type == 'cnn':
            self.char_pad_len = cmaps.size(2)
            # Sort by decreasing true word. sequence length
            wmap_lengths, word_sort_ind = wmap_lengths.sort(dim=0, descending=True)
            cmaps = cmaps[word_sort_ind]
            wmaps = wmaps[word_sort_ind]
            tmaps = tmaps[word_sort_ind]
            cmap_w_lengths = cmap_w_lengths[word_sort_ind]

            # CNN
            chars_embeds = self.char_embeds(cmaps)   # (batch, word_len, char_len, embed_dim)

            ## Creating Character level representation using Convolutional Neural Netowrk
            ## followed by a Maxpooling Layer
            chars_embeds = chars_embeds.view(-1, 1, self.char_pad_len, self.char_emb_dim)   # (batch * word_len, channel=1, char_len, embed_dim)

            chars_embeds = self.dropout(chars_embeds)

            chars_cnn_out3 = self.char_cnn3(chars_embeds)   # (batch * word_len, channel=char_feat_dim, char_len, embed_dim)

            chars_embeds = nn.functional.max_pool2d(chars_cnn_out3,
                kernel_size=(chars_cnn_out3.size(2), 1))    # (batch * word_len, channel=char_feat_dim, 1, 1)

            # recover shape
            chars_embeds = chars_embeds.view(self.batch_size, self.word_pad_len, self.char_feat_dim) # (batch, word_len, char_feat_dim)

            # Embedding look-up for words
            w = self.word_embeds(wmaps)  # (batch_size, word_pad_len, word_emb_dim)

            subword = chars_embeds

            # Concatenate word embeddings and sub-word features
            w = torch.cat((w, subword), dim=2)  # (batch_size, word_pad_len, word_emb_dim + 2 * char_feat_dim)
            w = self.dropout(w)

            # Pack padded sequence
            w = pack_padded_sequence(w, list(wmap_lengths), batch_first=True)  # packed sequence of word_emb_dim + 2 * char_feat_dim, with real sequence lengths

            # LSTM
            w, _ = self.word_blstm(w)  # packed sequence of word_rnn_dim, with real sequence lengths

            # Unpack packed sequence
            w, _ = pad_packed_sequence(w, batch_first=True)  # (batch_size, max_word_len_in_batch, word_rnn_dim)
            w = self.dropout(w)


            crf_scores = self.crf(w)  # (batch_size, max_word_len_in_batch, tagset_size, tagset_size)

        else:
            # Sort by decreasing true word. sequence length
            wmap_lengths, word_sort_ind = wmap_lengths.sort(dim=0, descending=True)
            wmaps = wmaps[word_sort_ind]
            tmaps = tmaps[word_sort_ind]

            # Embedding look-up for words
            w = self.word_embeds(wmaps)  # (batch_size, word_pad_len, word_emb_dim)

            w = self.dropout(w)

            # Pack padded sequence
            w = pack_padded_sequence(w, list(wmap_lengths), batch_first=True)  # packed sequence of word_emb_dim + 2 * char_feat_dim, with real sequence lengths

            # LSTM
            w, _ = self.word_blstm(w)  # packed sequence of word_rnn_dim, with real sequence lengths

            # Unpack packed sequence
            w, _ = pad_packed_sequence(w, batch_first=True)  # (batch_size, max_word_len_in_batch, word_rnn_dim)
            w = self.dropout(w)

            crf_scores = self.crf(w)  # (batch_size, max_word_len_in_batch, tagset_size, tagset_size)


        if self.training:
            if self.char_type == 'lstm':
                return crf_scores, lm_f_scores, lm_b_scores, wmaps, tmaps, wmap_lengths, word_sort_ind, char_sort_ind
            else:
                return crf_scores, None, None, wmaps, tmaps, wmap_lengths, word_sort_ind, None
        else:
            if self.char_type == 'lstm':
                return crf_scores, wmaps, tmaps, wmap_lengths, word_sort_ind, char_sort_ind
            else:
                return crf_scores, wmaps, tmaps, wmap_lengths, word_sort_ind, None

    def init_lstm(self, input_lstm):
        """
        Initialize lstm

        PyTorch weights parameters:

            weight_ih_l[k]: the learnable input-hidden weights of the k-th layer,
                of shape `(hidden_size * input_size)` for `k = 0`. Otherwise, the shape is
                `(hidden_size * hidden_size)`

            weight_hh_l[k]: the learnable hidden-hidden weights of the k-th layer,
                of shape `(hidden_size * hidden_size)`
        """

        # Weights init for forward layer
        for names in input_lstm._all_weights:
            for name in filter(lambda n: "weight" in n,  names):
                weight = getattr(input_lstm, name)
                sampling_range = np.sqrt(6.0 / (weight.size(0) + weight.size(1)))
                nn.init.uniform_(weight, -sampling_range, sampling_range)

        # Bias initialization steps
        if input_lstm.bias:
            for names in input_lstm._all_weights:
                for name in filter(lambda n: "bias" in n,  names):
                    bias = getattr(input_lstm, name)
                    # set all bias to 0
                    bias.data.zero_()
                    # except for forget gates which are set to zero
                    n = bias.size(0)
                    start, end = n//4, n//2
                    bias.data[start:end].fill_(1.)


class ViterbiLoss(nn.Module):
    """
    Viterbi Loss.
    """

    def __init__(self, tag_map):
        """
        :param tag_map: tag map
        """
        super(ViterbiLoss, self).__init__()
        self.tagset_size = len(tag_map)
        self.start_tag = tag_map['<start>']
        self.end_tag = tag_map['<end>']

    def forward(self, scores, targets, lengths):
        """
        Forward propagation.

        :param scores: CRF scores
        :param targets: true tags indices in unrolled CRF scores
        :param lengths: word sequence lengths
        :return: viterbi loss
        """

        batch_size = scores.size(0)
        word_pad_len = scores.size(1)

        # Gold score

        targets = targets.unsqueeze(2)
        scores_at_targets = torch.gather(scores.view(batch_size, word_pad_len, -1), 2, targets).squeeze(
            2)  # (batch_size, word_pad_len)

        # Everything is already sorted by lengths
        scores_at_targets, _, _, _ = pack_padded_sequence(scores_at_targets, lengths, batch_first=True)
        gold_score = scores_at_targets.sum()

        # All paths' scores

        # Create a tensor to hold accumulated sequence scores at each current tag
        scores_upto_t = torch.zeros(batch_size, self.tagset_size).to(device)

        for t in range(max(lengths)):
            batch_size_t = sum([l > t for l in lengths])  # effective batch size (sans pads) at this timestep
            if t == 0:
                scores_upto_t[:batch_size_t] = scores[:batch_size_t, t, self.start_tag, :]  # (batch_size, tagset_size)
            else:
                # We add scores at current timestep to scores accumulated up to previous timestep, and log-sum-exp
                # Remember, the cur_tag of the previous timestep is the prev_tag of this timestep
                # So, broadcast prev. timestep's cur_tag scores along cur. timestep's cur_tag dimension
                scores_upto_t[:batch_size_t] = log_sum_exp(
                    scores[:batch_size_t, t, :, :] + scores_upto_t[:batch_size_t].unsqueeze(2),
                    dim=1)  # (batch_size, tagset_size)

        # We only need the final accumulated scores at the <end> tag
        all_paths_scores = scores_upto_t[:, self.end_tag].sum()

        viterbi_loss = all_paths_scores - gold_score
        viterbi_loss = viterbi_loss / batch_size

        return viterbi_loss
