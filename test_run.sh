#!/bin/bash

python process_twitter_data_copy.py --train twitter.train.txt \
                                        --dev twitter.dev.txt \
                                        --test twitter.test.txt 


python vsl_gg.py \
    --debug 1 \
    --model hier \
    --data_file data/twitter1.0.data \
    --vocab_file twitter \
    --tag_file twitter/twitter_tagfile \
    --prior_file test_g \
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
    --f1_score True \
    --summarize 1


    


