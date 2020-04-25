import os
import pickle
import sys
import time

import torch
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.nn.utils.rnn import pack_padded_sequence

from datasets import WCDataset
from inference import ViterbiDecoder
from models import LM_LSTM_CRF, ViterbiLoss
from utils import *

from torch.utils.tensorboard import SummaryWriter

# visible gpus
os.environ['CUDA_VISIBLE_DEVICES'] = "0,2"

CONFIG_PATH = "./config.yml"
config = load_config(CONFIG_PATH)

# training config
PARAM_CONFIG_PATH = "./pos.yml"
param = load_config(PARAM_CONFIG_PATH)

# cpu / gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_metric_val = np.inf    # best metrics value
best_test = 0.          # best test value
epochs_since_improvement = 0
print_freq = 10  # print training or validation status every __ batches

# Writer will output to ./runs/ directory by default
writer = SummaryWriter()

def main():
    """
    Training and validation.
    """
    global best_metric_val, epochs_since_improvement, word_map, char_map, tag_map

    checkpoint = param.get('checkpoint', None)
    start_epoch = param['start_epoch']

    if param['task'] == 'pos':
        data_file = config['twitter_data']
    elif param['task'] == 'ner':
        data_file = config['eng_data']
    else:
        raise ValueError(f"Bad 'task' value: {param['task']}")

    train_words, train_tags = load_words_tags(data_file, 'train')
    val_words, val_tags = load_words_tags(data_file, 'dev')
    test_words, test_tags = load_words_tags(data_file, 'test')

    if param['caseless']:
        train_words = [[word.lower() for word in sentence] for sentence in train_words]
        val_words = [[word.lower() for word in sentence] for sentence in val_words]
        test_words = [[word.lower() for word in sentence] for sentence in test_words]

    # Initialize model or load checkpoint
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        word_map = checkpoint['word_map']
        lm_vocab_size = checkpoint['lm_vocab_size']
        tag_map = checkpoint['tag_map']
        char_map = checkpoint['char_map']
        start_epoch = checkpoint['epoch'] + 1
        best_metric_val = checkpoint['metric']
    else:
        word_map, char_map, tag_map = create_maps(train_words + val_words, train_tags + val_tags,
            param['min_word_freq'], param['min_char_freq'])  # create word, char, tag maps

        # load embeddings
        tmp_path = "tmp.npz"
        if os.path.exists(tmp_path):
            tmp_data = np.load(tmp_path, allow_pickle=True)
            embeddings, word_map, lm_vocab_size = tmp_data['embeddings'], tmp_data['word_map'], tmp_data['lm_vocab_size']
            embeddings = torch.from_numpy(embeddings)
            word_map = word_map.item()
        else:
            embeddings, word_map, lm_vocab_size = load_embeddings(config['embedding_path'], word_map, param['expand_vocab'])  # load pre-trained embeddings
            np.savez(tmp_path,
                embeddings=embeddings.cpu().numpy(),
                word_map=word_map,
                lm_vocab_size=lm_vocab_size,
            )

        # init model
        model = LM_LSTM_CRF(
            tagset_size=len(tag_map),
            tagset_map=tag_map,
            charset_size=len(char_map),
            char_emb_dim=param['char_emb_dim'],
            char_feat_dim=param['char_feat_dim'],
            char_rnn_layers=param['char_rnn_layers'],
            vocab_size=len(word_map),
            lm_vocab_size=lm_vocab_size,
            word_emb_dim=param['word_emb_dim'],
            word_rnn_dim=param['word_rnn_dim'],
            word_rnn_layers=param['word_rnn_layers'],
            dropout=param['dropout'],
            highway_layers=param['highway_layers'],
            char_type=param['char_type']
        ).to(device)
        model.init_word_embeddings(embeddings.to(device))  # initialize embedding layer with pre-trained embeddings
        model.fine_tune_word_embeddings(param['fine_tune_word_embeddings'])  # fine-tune
        optimizer = optim.SGD(params=filter(lambda p: p.requires_grad, model.parameters()),
            lr=param['lr'], momentum=param['momentum'])

    print(model)

    # Loss functions
    lm_criterion = nn.CrossEntropyLoss().to(device)
    crf_criterion = ViterbiLoss(tag_map).to(device)

    # Since the language model's vocab is restricted to in-corpus indices, encode training/val with only these!
    # word_map might have been expanded, and in-corpus words eliminated due to low frequency might still be added because
    # they were in the pre-trained embeddings
    temp_word_map = {k: v for k, v in word_map.items() if v <= word_map['<unk>']}
    train_inputs = create_input_tensors(train_words, train_tags, temp_word_map, char_map, tag_map)
    val_inputs = create_input_tensors(val_words, val_tags, temp_word_map, char_map, tag_map)
    test_inputs = create_input_tensors(test_words, test_tags, temp_word_map, char_map, tag_map)

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(WCDataset(*train_inputs), batch_size=param['batch_size'], shuffle=True,
                                            num_workers=param['workers'], pin_memory=False)
    val_loader = torch.utils.data.DataLoader(WCDataset(*val_inputs), batch_size=param['batch_size']*10,
                                            num_workers=param['workers'], pin_memory=False)
    test_loader = torch.utils.data.DataLoader(WCDataset(*test_inputs), batch_size=param['batch_size']*10,
                                            num_workers=param['workers'], pin_memory=False)

    # Viterbi decoder (to find accuracy during validation)
    vb_decoder = ViterbiDecoder(tag_map)

    # Epochs
    for epoch in range(start_epoch, param['epochs']):

        # One epoch's training
        train(
            train_loader=train_loader,
            model=model,
            lm_criterion=lm_criterion,
            crf_criterion=crf_criterion,
            optimizer=optimizer,
            epoch=epoch,
            vb_decoder=vb_decoder
        )

        # One epoch's validation
        vb_loss, f1, acc = validate(
            val_loader=val_loader,
            model=model,
            crf_criterion=crf_criterion,
            vb_decoder=vb_decoder,
            epoch=epoch
        )

        metric_val = vb_loss

        # Did loss improve?
        is_best = metric_val < best_metric_val
        best_metric_val = min(metric_val, best_metric_val)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since improvement: %d\n" % (epochs_since_improvement,))
            if epochs_since_improvement > param['early_stop']:
                # early stop
                break
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer, metric_val, word_map, char_map, tag_map, lm_vocab_size, is_best)

        # Decay learning rate every epoch
        adjust_learning_rate(optimizer, param['lr'] / (1 + (epoch + 1) * param['lr_decay']))

    # test
    checkpoint = torch.load(config['best_ckpt_path'])

    model = checkpoint['model']

    vb_loss, f1, acc = validate(test_loader, model, crf_criterion, vb_decoder, 0)
    print(f"** On test data: Loss = {vb_loss:.3f}, F1 = {f1:.3f}, Acc = {acc:.3f}")
    writer.close()


def train(train_loader, model, lm_criterion, crf_criterion, optimizer, epoch, vb_decoder):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param lm_criterion: cross entropy loss layer
    :param crf_criterion: viterbi loss layer
    :param optimizer: optimizer
    :param epoch: epoch number
    :param vb_decoder: viterbi decoder (to decode and find F1 score)
    """

    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time per batch
    data_time = AverageMeter()  # data loading time per batch
    ce_losses = AverageMeter()  # cross entropy loss
    vb_losses = AverageMeter()  # viterbi loss
    f1s = AverageMeter()  # f1 score
    accs = AverageMeter()  # accuracy

    start = time.time()

    # Batches
    for i, (wmaps, cmaps, cmaps_f, cmaps_b, cmarkers_f, cmarkers_b, tmaps, wmap_lengths,
            cmap_lengths, cmap_w_lengths) in enumerate(train_loader):

        data_time.update(time.time() - start)

        # prepare data
        max_word_len = max(wmap_lengths.tolist())
        if model.char_type == 'cnn':
            max_char_len = max([max(li) for li in cmap_w_lengths.tolist()])
            # Reduce batch's padded length to maximum in-batch sequence
            # This saves some compute on nn.Linear layers (RNNs are unaffected, since they don't compute over the pads)
            wmaps = wmaps[:, :max_word_len].to(device)
            cmaps = cmaps[:, :max_word_len, :max_char_len].to(device)
            cmap_w_lengths = cmap_w_lengths[:, :max_word_len].to(device)
            cmaps_f = None
            cmaps_b = None
            cmarkers_f = None
            cmarkers_b = None
            cmap_lengths = None
        else:
            max_char_len = max(cmap_lengths.tolist())

            # Reduce batch's padded length to maximum in-batch sequence
            # This saves some compute on nn.Linear layers (RNNs are unaffected, since they don't compute over the pads)
            wmaps = wmaps[:, :max_word_len].to(device)
            cmaps = None
            cmaps_f = cmaps_f[:, :max_char_len].to(device)
            cmaps_b = cmaps_b[:, :max_char_len].to(device)
            cmarkers_f = cmarkers_f[:, :max_word_len].to(device)
            cmarkers_b = cmarkers_b[:, :max_word_len].to(device)
            cmap_lengths = cmap_lengths.to(device)
            cmap_w_lengths = None

        tmaps = tmaps[:, :max_word_len].to(device)
        wmap_lengths = wmap_lengths.to(device)

        # Forward prop.
        forward_res = model(
            cmaps_f,
            cmaps_b,
            cmarkers_f,
            cmarkers_b,
            wmaps,
            cmaps,
            tmaps,
            wmap_lengths,
            cmap_lengths,
            cmap_w_lengths,
        )
        crf_scores, lm_f_scores, lm_b_scores, wmaps_sorted, tmaps_sorted, wmap_lengths_sorted, _, __ = forward_res

        # We don't predict the next word at the pads or <end> tokens
        # We will only predict at [dunston, checks, in] among [dunston, checks, in, <end>, <pad>, <pad>, ...]
        # So, prediction lengths are word sequence lengths - 1
        lm_lengths = wmap_lengths_sorted - 1
        lm_lengths = lm_lengths.tolist()

        # calc loss
        if model.char_type == 'lstm':
            # LM loss

            # Remove scores at timesteps we won't predict at
            # pack_padded_sequence is a good trick to do this (see dynamic_rnn.py, where we explore this)
            lm_f_scores, _, _, _ = pack_padded_sequence(lm_f_scores, lm_lengths, batch_first=True)
            lm_b_scores, _, _, _ = pack_padded_sequence(lm_b_scores, lm_lengths, batch_first=True)

            # For the forward sequence, targets are from the second word onwards, up to <end>
            # (timestep -> target) ...dunston -> checks, ...checks -> in, ...in -> <end>
            lm_f_targets = wmaps_sorted[:, 1:]
            lm_f_targets, _, _, _ = pack_padded_sequence(lm_f_targets, lm_lengths, batch_first=True)

            # For the backward sequence, targets are <end> followed by all words except the last word
            # ...notsnud -> <end>, ...skcehc -> dunston, ...ni -> checks
            lm_b_targets = torch.cat(
                [torch.LongTensor([word_map['<end>']] * wmaps_sorted.size(0)).unsqueeze(1).to(device), wmaps_sorted], dim=1)
            lm_b_targets, _, _, _ = pack_padded_sequence(lm_b_targets, lm_lengths, batch_first=True)

            # Calculate loss
            ce_loss = lm_criterion(lm_f_scores, lm_f_targets) + lm_criterion(lm_b_scores, lm_b_targets)
            vb_loss = crf_criterion(crf_scores, tmaps_sorted, wmap_lengths_sorted)
            loss = ce_loss + vb_loss
        else:
            # Calculate loss
            vb_loss = crf_criterion(crf_scores, tmaps_sorted, wmap_lengths_sorted)
            loss = vb_loss

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        if param['grad_clip'] is not None:
            clip_gradient(optimizer, param['grad_clip'])

        optimizer.step()

        # Viterbi decode to find accuracy / f1
        decoded = vb_decoder.decode(crf_scores.to(device), wmap_lengths_sorted.to(device))

        # Remove timesteps we won't predict at, and also <end> tags, because to predict them would be cheating
        decoded, _, _, _ = pack_padded_sequence(decoded, lm_lengths, batch_first=True)
        tmaps_sorted = tmaps_sorted % vb_decoder.tagset_size  # actual target indices (see create_input_tensors())
        tmaps_sorted, _, _, _ = pack_padded_sequence(tmaps_sorted, lm_lengths, batch_first=True)

        # F1
        f1 = f1_score(tmaps_sorted.to(device).cpu().numpy(), decoded.cpu().numpy(), average='macro')

        # acc
        pred = decoded.cpu().numpy()
        gt = tmaps_sorted.cpu().numpy()
        acc = sum(pred == gt) / len(pred)

        # Keep track of metrics
        if model.char_type == 'lstm':
            ce_losses.update(ce_loss.item(), sum(lm_lengths))
        vb_losses.update(vb_loss.item(), crf_scores.size(0))
        batch_time.update(time.time() - start)
        f1s.update(f1, sum(lm_lengths))
        accs.update(acc, len(pred))

        start = time.time()

        # Print training status
        if i % print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                # f'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                # f'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                f'CE Loss {ce_losses.val:.4f} ({ce_losses.avg:.4f})\t'
                f'VB Loss {vb_losses.val:.4f} ({vb_losses.avg:.4f})\t'
                f'F1 {f1s.val:.3f} ({f1s.avg:.3f})\t'
                f'Acc {accs.val:.3f} ({accs.avg:.3f})\t'
            )

    # end of epoch
    writer.add_scalar('Train Accuracy', accs.avg, epoch)
    writer.add_scalar('Train F1', f1s.avg, epoch)
    writer.add_scalar('Train VB Loss', vb_losses.avg, epoch)
    return


def validate(val_loader, model, crf_criterion, vb_decoder, epoch):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data
    :param model: model
    :param crf_criterion: viterbi loss layer
    :param vb_decoder: viterbi decoder
    :return: validation F1 score
    """
    model.eval()

    batch_time = AverageMeter()
    vb_losses = AverageMeter()
    f1s = AverageMeter()
    accs = AverageMeter()

    start = time.time()

    for i, (wmaps, cmaps, cmaps_f, cmaps_b, cmarkers_f, cmarkers_b, tmaps,
            wmap_lengths, cmap_lengths, cmap_w_lengths) in enumerate(val_loader):

        # prepare data
        max_word_len = max(wmap_lengths.tolist())
        if model.char_type == 'cnn':
            max_char_len = max([max(li) for li in cmap_w_lengths.tolist()])

            # Reduce batch's padded length to maximum in-batch sequence
            # This saves some compute on nn.Linear layers (RNNs are unaffected, since they don't compute over the pads)
            wmaps = wmaps[:, :max_word_len].to(device)
            cmaps = cmaps[:, :max_word_len, :max_char_len].to(device)
            cmap_w_lengths = cmap_w_lengths[:, :max_word_len].to(device)
            cmaps_f = None
            cmaps_b = None
            cmarkers_f = None
            cmarkers_b = None
            cmap_lengths = None
        else:
            max_char_len = max(cmap_lengths.tolist())

            # Reduce batch's padded length to maximum in-batch sequence
            # This saves some compute on nn.Linear layers (RNNs are unaffected, since they don't compute over the pads)
            wmaps = wmaps[:, :max_word_len].to(device)
            cmaps = None
            cmaps_f = cmaps_f[:, :max_char_len].to(device)
            cmaps_b = cmaps_b[:, :max_char_len].to(device)
            cmarkers_f = cmarkers_f[:, :max_word_len].to(device)
            cmarkers_b = cmarkers_b[:, :max_word_len].to(device)
            cmap_lengths = cmap_lengths.to(device)
            cmap_w_lengths = None

        tmaps = tmaps[:, :max_word_len].to(device)
        wmap_lengths = wmap_lengths.to(device)

        # Forward prop.
        forward_res = model(
            cmaps_f,
            cmaps_b,
            cmarkers_f,
            cmarkers_b,
            wmaps,
            cmaps,
            tmaps,
            wmap_lengths,
            cmap_lengths,
            cmap_w_lengths,
        )
        crf_scores, wmaps_sorted, tmaps_sorted, wmap_lengths_sorted, _, __ = forward_res

        # Viterbi / CRF layer loss
        vb_loss = crf_criterion(crf_scores, tmaps_sorted, wmap_lengths_sorted)

        # Viterbi decode to find accuracy / f1
        decoded = vb_decoder.decode(crf_scores.to(device), wmap_lengths_sorted.to(device))

        # Remove timesteps we won't predict at, and also <end> tags, because to predict them would be cheating
        decoded, _, _, _ = pack_padded_sequence(decoded, (wmap_lengths_sorted - 1).tolist(), batch_first=True)
        tmaps_sorted = tmaps_sorted % vb_decoder.tagset_size  # actual target indices (see create_input_tensors())
        tmaps_sorted, _, _, _ = pack_padded_sequence(tmaps_sorted, (wmap_lengths_sorted - 1).tolist(), batch_first=True)

        # f1
        f1 = f1_score(tmaps_sorted.cpu().numpy(), decoded.cpu().numpy(), average='macro')

        # acc
        pred = decoded.cpu().numpy()
        gt = tmaps_sorted.cpu().numpy()
        acc = sum(pred == gt) / len(pred)

        # Keep track of metrics
        vb_losses.update(vb_loss.item(), crf_scores.size(0))
        f1s.update(f1, sum((wmap_lengths_sorted - 1).tolist()))
        batch_time.update(time.time() - start)
        accs.update(acc, len(pred))

        start = time.time()

        if i % print_freq == 0:
            print(f'Validation: [{i}/{len(val_loader)}]\t'
                # f'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                # f'VB Loss {vb_losses.val:.4f} ({vb_losses.avg:.4f})\t'
                f'F1 Score {f1s.val:.3f} ({f1s.avg:.3f})\t'
                f'Acc {accs.val:.3f} ({accs.avg:.3f})'
            )

    writer.add_scalar('Valid Accuracy', accs.avg, epoch)
    writer.add_scalar('Valid F1', f1s.avg, epoch)
    writer.add_scalar('Valid VB Loss', vb_losses.avg, epoch)

    print(f'* LOSS - {vb_losses.avg:.3f}, F1 SCORE - {f1s.avg:.3f}, Acc - {accs.avg:.3f}\n')

    # return f1s.avg if param['task'] == 'ner' else accs.avg
    return vb_losses.avg, f1s.avg, accs.avg


if __name__ == '__main__':
    main()
