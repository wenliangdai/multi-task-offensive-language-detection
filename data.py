import os
import pandas as pd
import numpy as np
import emoji
import nltk
import wordsegment
from embeddings import GloveEmbedding
from nltk.tokenize import word_tokenize
# from vocabulary import Vocab
from config import OLID_PATH # SAVE_DIR, PAD_TOKEN, SEP_TOEKN
from utils import save, load, pad_sents, sort_sents, get_lens, get_mask, truncate_sents
from transformers import BertTokenizer

# Uncomment this line if you haven't download nltk packages
# nltk.download()
wordsegment.load()

def read_file(filepath: str):
    df = pd.read_csv(filepath, sep='\t')

    ids = np.array(df['id'].values)
    tweets = np.array(df['tweet'].values)

    # Process tweets
    tweets = emoji2word(tweets)
    tweets = replace_rare_words(tweets)
    tweets = remove_replicates(tweets)
    tweets = segment_hashtag(tweets)
    tweets = remove_useless_punctuation(tweets)
    tweets = np.array(tweets)

    # with open('temp.txt', 'w') as f:
    #     for t in tweets:
    #         f.write(t + '\n')
    # exit(1)

    label_a = np.array(df['subtask_a'].values)
    label_b = np.array(df['subtask_b'].values)
    label_c = np.array(df['subtask_c'].values)
    nums = len(df)

    return nums, ids, tweets, label_a, label_b, label_c

def read_test_file(task, truncate=-1):
    df1 = pd.read_csv(os.path.join(OLID_PATH, 'testset-level' + task + '.tsv'), sep='\t')
    df2 = pd.read_csv(os.path.join(OLID_PATH, 'labels-level' + task + '.csv'), sep=',')
    ids = np.array(df1['id'].values)
    tweets = np.array(df1['tweet'].values)
    labels = np.array(df2['label'].values)
    nums = len(df1)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    token_ids = [tokenizer.encode(text=tweets[i], add_special_tokens=True) for i in range(nums)]
    mask = np.array(get_mask(token_ids))
    token_ids = np.array(pad_sents(token_ids, tokenizer.pad_token_id))

    if truncate > 0:
        token_ids = truncate_sents(token_ids, truncate)
        mask = truncate_sents(mask, truncate)

    return ids, token_ids, mask, labels

def emoji2word(sents):
    return [emoji.demojize(sent) for sent in sents]

def remove_useless_punctuation(sents):
    for i, sent in enumerate(sents):
        sent = sent.replace(':', ' ')
        sent = sent.replace('_', ' ')
        sent = sent.replace('...', ' ')
        sents[i] = sent
    return sents

def remove_replicates(sents):
    # if there are multiple `@USER` tokens in a tweet, replace it with `@USERS`
    # because some tweets contain so many `@USER` which may cause redundant
    for i, sent in enumerate(sents):
        if sent.find('@USER') != sent.rfind('@USER'):
            sents[i] = sent.replace('@USER', '')
            sents[i] = '@USERS ' + sents[i]
    return sents

def replace_rare_words(sents):
    rare_words = {
        'URL': 'http'
    }
    for i, sent in enumerate(sents):
        for w in rare_words.keys():
            sents[i] = sent.replace(w, rare_words[w])
    return sents

def segment_hashtag(sents):
    # E.g. '#LunaticLeft' => 'lunatic left'
    for i, sent in enumerate(sents):
        sent_tokens = sent.split(' ')
        for j, t in enumerate(sent_tokens):
            if t.find('#') == 0:
                sent_tokens[j] = ' '.join(wordsegment.segment(t))
        sents[i] = ' '.join(sent_tokens)
    return sents

def bert_all_tasks(filepath: str, truncate=-1):
    nums, ids, tweets, label_a, label_b, label_c = read_file(filepath)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    token_ids = [tokenizer.encode(text=tweets[i], add_special_tokens=True) for i in range(nums)]
    mask = np.array(get_mask(token_ids))
    token_ids = np.array(pad_sents(token_ids, tokenizer.pad_token_id))

    if truncate > 0:
        token_ids = truncate_sents(token_ids, truncate)
        mask = truncate_sents(mask, truncate)

    return ids, token_ids, mask, label_a, label_b, label_c

def bert_task_a(filepath: str, truncate=-1):
    nums, ids, tweets, label_a, _, _ = read_file(filepath)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    token_ids = [tokenizer.encode(text=tweets[i], add_special_tokens=True) for i in range(nums)]
    mask = np.array(get_mask(token_ids))
    token_ids = np.array(pad_sents(token_ids, tokenizer.pad_token_id))

    if truncate > 0:
        token_ids = truncate_sents(token_ids, truncate)
        mask = truncate_sents(mask, truncate)

    return ids, token_ids, mask, label_a

def bert_task_b(filepath: str, truncate=-1):
    nums, ids, tweets, _, label_b, _ = read_file(filepath)
    # Only part of the tweets are useful for task b
    useful = label_b != 'NULL'
    ids = ids[useful]
    tweets = tweets[useful]
    label_b = label_b[useful]
    nums = len(label_b)
    # Tokenize
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    token_ids = [tokenizer.encode(text=tweets[i], add_special_tokens=True) for i in range(nums)]
    # Get mask
    mask = np.array(get_mask(token_ids))
    # Pad tokens
    token_ids = np.array(pad_sents(token_ids, tokenizer.pad_token_id))

    if truncate > 0:
        token_ids = truncate_sents(token_ids, truncate)
        mask = truncate_sents(mask, truncate)

    return ids, token_ids, mask, label_b

def bert_task_c(filepath: str, truncate=-1):
    nums, ids, tweets, _, _, label_c = read_file(filepath)
    # Only part of the tweets are useful for task c
    useful = label_c != 'NULL'
    ids = ids[useful]
    tweets = tweets[useful]
    label_c = label_c[useful]
    nums = len(label_c)
    # Tokenize
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    token_ids = [tokenizer.encode(text=tweets[i], add_special_tokens=True) for i in range(nums)]
    # Get mask
    mask = np.array(get_mask(token_ids))
    # Pad tokens
    token_ids = np.array(pad_sents(token_ids, tokenizer.pad_token_id))

    if truncate > 0:
        token_ids = truncate_sents(token_ids, truncate)
        mask = truncate_sents(mask, truncate)

    return ids, token_ids, mask, label_c
