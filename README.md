# BERT-Based Multi-Task Learning for Offensive Language Detection

<img src="img/pytorch-logo-dark.png" width="10%"/> [![](https://img.shields.io/badge/python-3.5+-orange.svg)](https://www.python.org/downloads/) [![CC BY 4.0][cc-by-shield]][cc-by]


<img align="right" src="img/HKUST.jpg" width="15%"/>

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

Paper accepted at the [SemEval-2020](https://www.aclweb.org/anthology/events/semeval-2020/) (COLING 2020):

**Kungfupanda at SemEval-2020 Task 12: BERT-Based Multi-Task Learning for Offensive Language Detection**, by **[Wenliang Dai*](https://wenliangdai.github.io/)**, Tiezheng Yu*, [Zihan Liu](https://zliucr.github.io/), Pascale Fung.

[[ACL Anthology](https://www.aclweb.org/anthology/2020.semeval-1.272/)][[ArXiv](https://arxiv.org/abs/2004.13432)][[Semantic Scholar](https://www.semanticscholar.org/paper/Kungfupanda-at-SemEval-2020-Task-12%3A-BERT-Based-for-Dai-Yu/fa46a8c826fc456923fba090e6c837d0813c98f9)]

If your work is inspired by our paper, or you use any code snippets in this repo, please cite this paper, the BibTex is shown below:

<pre>
@article{Dai2020KungfupandaAS,
  title={Kungfupanda at SemEval-2020 Task 12: BERT-Based Multi-Task Learning for Offensive Language Detection},
  author={Wenliang Dai and Tiezheng Yu and Zihan Liu and Pascale Fung},
  journal={ArXiv},
  year={2020},
  volume={abs/2004.13432}
}
</pre>

## Abstract

Nowadays, offensive content in social media has become a serious problem. 
Transfer learning and multi-task learning are two major techniques that are widely employed in machine learning fields. With transfer learning, we can effectively learn a related problem from a well pre-trained model. In addition, one of the benefits of using multi-task learning is to have more supervision signals and a better generalization ability. In the task of Multilingual Offensive Language Identification in Social Media, we propose to use both techniques to make use of pre-trained feature representations and better leverage the information in the hierarchical dataset. Our contribution is two-fold. Firstly, we propose a multi-task transfer learning model for this problem and we provide an empirical analysis to explain why this method is very effective. Secondly, the model achieves a performance (91.51\% F1) comparable to the first place (92.23\% F1) in the competition with only the OLID dataset.

## Dataset

1. [Offensive Language Identification Dataset (OLID)](https://sites.google.com/site/offensevalsharedtask/olid)


## Requirements

1. Python 3.5 +
2. PyTorch 1.3 +
3. Huggingface transformers 2.x
4. We use one GTX 1080Ti to train

## Command Line Arguments

```
usage: train.py [-h] -bs BATCH_SIZE -lr LEARNING_RATE [-wd WEIGHT_DECAY] -ep
                EPOCHS [-tr TRUNCATE] [-pa PATIENCE] [-cu CUDA] -ta TASK -mo
                MODEL [-ms MODEL_SIZE] [-cl] [-fr FREEZE]
                [-lw LOSS_WEIGHTS [LOSS_WEIGHTS ...]] [-sc] [-se SEED]
                [--ckpt CKPT] [-ad ATTENTION_DROPOUT] [-hd HIDDEN_DROPOUT]
                [-dr DROPOUT] [-nl NUM_LAYERS] [-hs HIDDEN_SIZE]
                [-hcm HIDDEN_COMBINE_METHOD]

BERT-Based Multi-Task Learning for Offensive Language Detection

optional arguments:
  -h, --help            show this help message and exit
  -bs BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size
  -lr LEARNING_RATE, --learning-rate LEARNING_RATE
                        Learning rate
  -wd WEIGHT_DECAY, --weight-decay WEIGHT_DECAY
                        Weight decay
  -ep EPOCHS, --epochs EPOCHS
                        Number of epochs
  -tr TRUNCATE, --truncate TRUNCATE
                        Truncate the sequence length to
  -pa PATIENCE, --patience PATIENCE
                        Patience to stop training
  -cu CUDA, --cuda CUDA
                        Cude device number
  -ta TASK, --task TASK
                        Which subtask to run
  -mo MODEL, --model MODEL
                        Which model to use
  -ms MODEL_SIZE, --model-size MODEL_SIZE
                        Which size of model to use
  -cl, --clip           Use clip to gradients
  -fr FREEZE, --freeze FREEZE
                        Freeze the embedding layer or not to use less GPU
                        memory
  -lw LOSS_WEIGHTS [LOSS_WEIGHTS ...], --loss-weights LOSS_WEIGHTS [LOSS_WEIGHTS ...]
                        Weights for all losses
  -sc, --scheduler      Use scheduler to optimizer
  -se SEED, --seed SEED
                        Random seed
  --ckpt CKPT
  -ad ATTENTION_DROPOUT, --attention-dropout ATTENTION_DROPOUT
                        transformer attention dropout
  -hd HIDDEN_DROPOUT, --hidden-dropout HIDDEN_DROPOUT
                        transformer hidden dropout
  -dr DROPOUT, --dropout DROPOUT
                        dropout
  -nl NUM_LAYERS, --num-layers NUM_LAYERS
                        num of layers of LSTM
  -hs HIDDEN_SIZE, --hidden-size HIDDEN_SIZE
                        hidden vector size of LSTM
  -hcm HIDDEN_COMBINE_METHOD, --hidden-combine-method HIDDEN_COMBINE_METHOD
                        how to combbine hidden vectors in LSTM
```

## Usage Examples

Train single-task model for subtask-A

```console
python train.py -bs=32 -lr=3e-6 -ep=20 -pa=3 --model=bert --task=a --clip --cuda=1
```

Train multi-task model

```console
python train.py -bs=32 -lr=3e-6 -ep=20 -pa=3 --model=bert --task=all --clip --loss-weights 0.4 0.3 0.3 --cuda=1
```
