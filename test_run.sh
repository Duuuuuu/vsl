#!/bin/bash

rm -r ud/ud_prior
#test ud, vsl with unlabel
python vsl_gg.py \
    --debug 1 \
    --model hier \
    --data_file ud/ud0.2_.data \
    --vocab_file ud \
    --tag_file ud/ud_tagfile\
    --prior_file ud/ud_prior \
    --embed_file ud/es.bin \
    --embed_type ud \
    --n_iter 30000 \
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
    --use_unlabel True \
    --unlabel_file ud/ud0.2_unlabel_.data \
    --model_name ud-vsl-su

rm -r ud/ud_prior

#test ud, vsl-crf without unlabel
python vsl_gg_crf.py \
    --debug 1 \
    --model hier \
    --data_file ud/ud0.2_crf.data \
    --vocab_file ud \
    --tag_file ud/ud_tagfile_crf \
    --prior_file ud/ud_prior \
    --embed_file ud/es.bin \
    --embed_type ud \
    --n_iter 30000 \
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
    --model_name ud-vsl-crf-s

rm -r ud/ud_prior

#test ud, vsl-crf without unlabel
python vsl_gg_crf.py \
    --debug 1 \
    --model hier \
    --data_file ud/ud0.2_crf.data \
    --vocab_file ud \
    --tag_file ud/ud_tagfile_crf \
    --prior_file ud/ud_prior \
    --embed_file ud/es.bin \
    --embed_type ud \
    --n_iter 30000 \
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
    --use_unlabel True \
    --unlabel_file ud/ud0.2_unlabel_crf.data \
    --model_name ud-vsl-crf-su
    
rm -r ud/ud_prior  

#test ner,vsl, without unlabel
rm -r ner/ner_prior

python vsl_gg.py \
    --debug 1 \
    --model hier \
    --data_file ner/ner0.1_.data \
    --vocab_file ner \
    --tag_file ner/ner_tagfile \
    --prior_file ner/ner_prior \
    --embed_file ner/glove_vocab.txt \
    --embed_type glove \
    --n_iter 30000 \
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
    --model_name ner-vsl-s 
   


#test ner,vsl, with unlabel
rm -r ner/ner_prior

python vsl_gg.py \
    --debug 1 \
    --model hier \
    --data_file ner/ner0.1_.data \
    --vocab_file ner \
    --tag_file ner/ner_tagfile \
    --prior_file ner/ner_prior \
    --embed_file ner/glove_vocab.txt \
    --embed_type glove \
    --n_iter 30000 \
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
    --unlabel_file ner/ner0.1_unlabel_.data \
    --model_name ner-vsl-su 

#test ner,vsl-crf, without unlabel
rm -r ner/ner_prior

python vsl_gg_crf.py \
    --debug 1 \
    --model hier \
    --data_file ner/ner0.1_crf.data \
    --vocab_file ner \
    --tag_file ner/ner_tagfile_crf \
    --prior_file ner/ner_prior \
    --embed_file ner/glove_vocab.txt \
    --embed_type glove \
    --n_iter 30000 \
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
    --model_name ner-vsl-crf-s 


#test ner,vsl-crf, with unlabel
rm -r ner/ner_prior

python vsl_gg_crf.py \
    --debug 1 \
    --model hier \
    --data_file ner/ner0.1_crf.data \
    --vocab_file ner \
    --tag_file ner/ner_tagfile_crf \
    --prior_file ner/ner_prior \
    --embed_file ner/glove_vocab.txt \
    --embed_type glove \
    --n_iter 30000 \
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
    --unlabel_file ner/ner0.1_unlabel_crf.data \
    --model_name ner-vsl-crf-su

rm -r ner/ner_prior


