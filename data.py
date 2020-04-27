import os
import pandas as pd
import numpy as np
import emoji
import wordsegment
from config import OLID_PATH
from utils import pad_sents, get_mask, get_lens

wordsegment.load()

def read_file(filepath: str):
    df = pd.read_csv(filepath, sep='\t', keep_default_na=False)

    ids = np.array(df['id'].values)
    tweets = np.array(df['tweet'].values)

    # Process tweets
    tweets = process_tweets(tweets)

    label_a = np.array(df['subtask_a'].values)
    label_b = df['subtask_b'].values
    label_c = np.array(df['subtask_c'].values)
    nums = len(df)

    return nums, ids, tweets, label_a, label_b, label_c

def read_test_file(task, tokenizer, truncate=512):
    df1 = pd.read_csv(os.path.join(OLID_PATH, 'testset-level' + task + '.tsv'), sep='\t')
    df2 = pd.read_csv(os.path.join(OLID_PATH, 'labels-level' + task + '.csv'), sep=',')
    ids = np.array(df1['id'].values)
    tweets = np.array(df1['tweet'].values)
    labels = np.array(df2['label'].values)
    nums = len(df1)

    # Process tweets
    tweets = process_tweets(tweets)

    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    token_ids = [tokenizer.encode(text=tweets[i], add_special_tokens=True, max_length=truncate) for i in range(nums)]
    mask = np.array(get_mask(token_ids))
    lens = get_lens(token_ids)
    token_ids = np.array(pad_sents(token_ids, tokenizer.pad_token_id))

    return ids, token_ids, lens, mask, labels

def read_test_file_all(tokenizer, truncate=512):
    df = pd.read_csv(os.path.join(OLID_PATH, 'testset-levela.tsv'), sep='\t')
    df_a = pd.read_csv(os.path.join(OLID_PATH, 'labels-levela.csv'), sep=',')
    ids = np.array(df['id'].values)
    tweets = np.array(df['tweet'].values)
    label_a = np.array(df_a['label'].values)
    nums = len(df)

    # Process tweets
    tweets = process_tweets(tweets)

    df_b = pd.read_csv(os.path.join(OLID_PATH, 'labels-levelb.csv'), sep=',')
    df_c = pd.read_csv(os.path.join(OLID_PATH, 'labels-levelc.csv'), sep=',')
    label_data_b = dict(zip(df_b['id'].values, df_b['label'].values))
    label_data_c = dict(zip(df_c['id'].values, df_c['label'].values))
    label_b = [label_data_b[id] if id in label_data_b.keys() else 'NULL' for id in ids]
    label_c = [label_data_c[id] if id in label_data_c.keys() else 'NULL' for id in ids]

    token_ids = [tokenizer.encode(text=tweets[i], add_special_tokens=True, max_length=truncate) for i in range(nums)]
    mask = np.array(get_mask(token_ids))
    lens = get_lens(token_ids)
    token_ids = np.array(pad_sents(token_ids, tokenizer.pad_token_id))

    return ids, token_ids, lens, mask, label_a, label_b, label_c

def process_tweets(tweets):
    # Process tweets
    tweets = emoji2word(tweets)
    tweets = replace_rare_words(tweets)
    tweets = remove_replicates(tweets)
    tweets = segment_hashtag(tweets)
    tweets = remove_useless_punctuation(tweets)
    tweets = np.array(tweets)
    return tweets

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

def all_tasks(filepath: str, tokenizer, truncate=512):
    nums, ids, tweets, label_a, label_b, label_c = read_file(filepath)
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    token_ids = [tokenizer.encode(text=tweets[i], add_special_tokens=True, max_length=truncate) for i in range(nums)]
    mask = np.array(get_mask(token_ids))
    lens = get_lens(token_ids)
    token_ids = np.array(pad_sents(token_ids, tokenizer.pad_token_id))

    return ids, token_ids, lens, mask, label_a, label_b, label_c

def task_a(filepath: str, tokenizer, truncate=512):
    nums, ids, tweets, label_a, _, _ = read_file(filepath)
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    token_ids = [tokenizer.encode(text=tweets[i], add_special_tokens=True, max_length=truncate) for i in range(nums)]
    mask = np.array(get_mask(token_ids))
    lens = get_lens(token_ids)
    token_ids = np.array(pad_sents(token_ids, tokenizer.pad_token_id))

    return ids, token_ids, lens, mask, label_a

def task_b(filepath: str, tokenizer, truncate=512):
    nums, ids, tweets, _, label_b, _ = read_file(filepath)
    # Only part of the tweets are useful for task b

    useful = label_b != 'NULL'
    ids = ids[useful]
    tweets = tweets[useful]
    label_b = label_b[useful]

    nums = len(label_b)
    # Tokenize
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    token_ids = [tokenizer.encode(text=tweets[i], add_special_tokens=True, max_length=truncate) for i in range(nums)]
    # Get mask
    mask = np.array(get_mask(token_ids))
    # Get lengths
    lens = get_lens(token_ids)
    # Pad tokens
    token_ids = np.array(pad_sents(token_ids, tokenizer.pad_token_id))

    return ids, token_ids, lens, mask, label_b

def task_c(filepath: str, tokenizer, truncate=512):
    nums, ids, tweets, _, _, label_c = read_file(filepath)
    # Only part of the tweets are useful for task c
    useful = label_c != 'NULL'
    ids = ids[useful]
    tweets = tweets[useful]
    label_c = label_c[useful]
    nums = len(label_c)
    # Tokenize
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    token_ids = [tokenizer.encode(text=tweets[i], add_special_tokens=True, max_length=truncate) for i in range(nums)]
    # Get mask
    mask = np.array(get_mask(token_ids))
    # Get lengths
    lens = get_lens(token_ids)
    # Pad tokens
    token_ids = np.array(pad_sents(token_ids, tokenizer.pad_token_id))

    return ids, token_ids, lens, mask, label_c
