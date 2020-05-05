#!/bin/bash

python vsl_gg_crf.py \
    --debug 1 \
    --model hier \
    --data_file ud/ud0.2_crf.data \
    --vocab_file ud \
    --tag_file ud/ud_tagfile_crf \
    --prior_file ud/ud_prior \
    --embed_file ud/es.bin \
    --embed_type ud \
    --n_iter 200 \
    --save_prior 1 \
    --train_emb 0 \
    --tie_weights 1 \
    --embed_dim 100 \
    --latent_z_size 50 \
    --update_freq_label 1 \
    --update_freq_unlabel 1 \
    --rnn_size 100 \
    --char_embed_dim 50 \
    --char_hidden_size 100 \
    --mlp_layer 2 \
    --mlp_hidden_size 100 \
    --learning_rate 1e-3 \
    --vocab_size 100000 \
    --batch_size 10 \
    --kl_anneal_rate 1e-4 \
    --print_every 100 \
    --eval_every 100 \
    --vb_temp 0.5 \
    --f1_score True \
    --summarize 1 \
    --use_unlabel True\
    --unlabel_file ud/ud0.2_unlabel_crf.data



python process_ner_data.py --train eng.train \
                          --dev eng.testa \
                          --test eng.testb \
                          --ratio 0.1
                          --crf True

python process_ud_data.py --train en-ud-train.conllu \
                          --dev en-ud-dev.conllu \
                          --test en-ud-test.conllu \
                          --ratio 0.2 \
                          --crf True

python process_twitter_data.py --train twitter.train.txt \
                               --dev twitter.dev.txt \
                               --test twitter.test.txt \
                               --ratio 0.3 \
                               --crf True


python vsl_gg_crf.py \
    --debug 1 \
    --model hier \
    --data_file twitter/twitter0.3_crf.data \
    --vocab_file twitter \
    --tag_file twitter/twitter_tagfile_crf \
    --prior_file twitter/twitter_prior \
    --embed_file twitter/twitter_wordvects \
    --n_iter 200 \
    --save_prior 1 \
    --train_emb 0 \
    --tie_weights 1 \
    --embed_dim 100 \
    --latent_z_size 50 \
    --update_freq_label 1 \
    --update_freq_unlabel 1 \
    --rnn_size 100 \
    --char_embed_dim 50 \
    --char_hidden_size 100 \
    --mlp_layer 2 \
    --mlp_hidden_size 100 \
    --learning_rate 1e-3 \
    --vocab_size 100000 \
    --batch_size 10 \
    --kl_anneal_rate 1e-4 \
    --print_every 100 \
    --eval_every 100 \
    --vb_temp 0.5 \
    --f1_score True \
    --summarize 1 \
    --use_unlabel True\
    --unlabel_file twitter/twitter0.3_unlabel_crf.data


python vsl_gg_crf.py \
    --debug 1 \
    --model hier \
    --data_file ner/ner0.1_crf.data \
    --vocab_file ner \
    --tag_file ner/ner_tagfile_crf \
    --prior_file ner/ner_prior \
    --embed_file ner/glove_vocab.txt \
    --embed_type glove \
    --n_iter 200 \
    --save_prior 1 \
    --train_emb 0 \
    --tie_weights 1 \
    --embed_dim 100 \
    --latent_z_size 50 \
    --update_freq_label 1 \
    --update_freq_unlabel 1 \
    --rnn_size 100 \
    --char_embed_dim 50 \
    --char_hidden_size 100 \
    --mlp_layer 2 \
    --mlp_hidden_size 100 \
    --learning_rate 1e-3 \
    --vocab_size 100000 \
    --batch_size 10 \
    --kl_anneal_rate 1e-4 \
    --print_every 100 \
    --eval_every 100 \
    --vb_temp 0.5 \
    --f1_score True \
    --summarize 1 \
    --use_unlabel True\
    --unlabel_file ner/ner0.1_unlabel_crf.data
                    