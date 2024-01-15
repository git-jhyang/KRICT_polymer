# KRICT Polymer
Predicting properties of copolymer using machine learning models. 

## pretrain.py
Train neural network models via QM9 dataset.

```bash
python pretrain.py pretrain.json concat/cg_32.128_4.1.json --verbose
```

## train.py
Train neural network models via copolymer dataset. 
```bash
python train.py scratch.json hp_opt/b04_g75_s25.json network/cg_64.128_6.2.json --verbose
python train.py finetune.json hp_opt/b04_g75_s25.json network/cg_64.128_6.2.json --verbose
```

## bayesian.py
Train neural network models via copolymer dataset. Model performance is optimized via Bayesian optimizer (learning rate only).

```bash
python bayesian.py bayesian.json scratch.json hp_opt/b04_g75_s25.json network/cg_64.128_6.2.json --verbose
```

## tg_prediction.py
Predict glass transition temperature of copolymer.

```bash
python tg_prediction.py ./data/example_copolymer.csv
```

