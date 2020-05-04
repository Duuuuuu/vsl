# VSL

A PyTorch implementation of "[Variational Sequential Labelers for Semi-Supervised Learning](http://ttic.uchicago.edu/~mchen/papers/mchen+etal.emnlp18.pdf)" (EMNLP 2018)


## Prerequisites

- Python 3.5
- PyTorch 0.3.0
- Scikit-Learn
- NumPy

## Data and Pretrained Embeddings

Download: [Twitter](https://code.google.com/archive/p/ark-tweet-nlp/downloads), [Universal Dependencies](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-1827?show=full), [Embeddings (for Twitter and UD)](https://drive.google.com/drive/folders/1oie43_thsbhhoUsOHlkyKj2iMpFNOrgA?usp=sharing)

Run `process_{ner,twitter,ud}_data.py` first to generate `*.pkl` files and then use it as input for `vsl_{g,gg}.py`.


### Data Preperation
save the processed files in the folder data/ 

prepare the targeted folder ud/ or twitter/ or ner/

The suggested ratio for NER is 0.1, for UD is 0.2, for twitter is 0.3.

```
python process_ner_data.py --train eng.train \
                          --dev eng.testa \
                          --test eng.testb \
                          --ratio 0.1
                          --crf True
```



### Train and Evaluate the model

To add crf for training, use files with suffix `_crf`

Please differentiate prior names for different datasets

To add unlabeled files:
```
--use_unlabel True
--unlabel_file ner/ner0.1_unlabel_.data
```

see `test_run.sh` for commands