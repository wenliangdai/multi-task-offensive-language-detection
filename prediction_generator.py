## Thsi is for generate the test prediction labels from model

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from data import task_a, task_b, task_c, all_tasks, read_test_file, read_test_file_all, process_tweets, get_mask, get_lens, pad_sents
from config import OLID_PATH
from cli import get_args
from utils import load
from tqdm import tqdm
import csv
# from utils import get_loss_weight
from datasets import HuggingfaceDataset, HuggingfaceMTDataset, ImbalancedDatasetSampler
from models.bert import BERT, RoBERTa, MTModel, BERT_LSTM
from models.gated import GatedModel
from models.mtl import MTL_Transformer_LSTM, MTL_Transformer_LSTM_gate
from transformers import BertTokenizer, RobertaTokenizer#, WarmupCosineSchedule
from trainer import Trainer

def read_test_data(tokenizer, test_file):
    df1 = pd.read_csv(test_file, sep='\t')
    ids = np.array(df1['id'].values)
    tweets = np.array(df1['tweet'].values)
    nums = len(df1)
    # Process tweets
    tweets = process_tweets(tweets)
    
    token_ids = [tokenizer.encode(text=tweets[i], add_special_tokens=True) for i in range(nums)]
    mask = np.array(get_mask(token_ids))
    lens = get_lens(token_ids)
    token_ids = np.array(pad_sents(token_ids, tokenizer.pad_token_id))
    return ids, token_ids, mask


class TestDataset(Dataset):
    def __init__(self, ids, input_ids, mask):
        self.ids =  torch.tensor(ids)
        self.input_ids = torch.tensor(input_ids)
        self.mask = torch.tensor(mask, dtype=torch.float32)

    def __len__(self):
        return self.ids.shape[0]

    def __getitem__(self, idx):
        ids = self.ids[idx]
        input_ids = self.input_ids[idx]
        length = self.input_ids[idx]
        mask = self.mask[idx]
        return ids, input_ids, length, mask

    
if __name__ == '__main__':
    
    # Get command line arguments
    args = get_args()
    task = args['task']
    model_name = args['model']
    model_size = args['model_size']
    truncate = args['truncate']
    epochs = args['epochs']
    lr = args['learning_rate']
    wd = args['weight_decay']
    bs = args['batch_size']
    print_iter = args['print_iter']
    patience = args['patience']
    
    # Fix seed for reproducibility
    seed = args['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = args['cuda']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    num_labels = 3 if task == 'c' else 2
    
    # Set tokenizer for different models

    if model_name == 'bert':
        if task == 'all':
            model = MTL_Transformer_LSTM_gate(model_name, model_size, args=args)
        else:
            model = BERT(model_size, args=args, num_labels=num_labels)
        tokenizer = BertTokenizer.from_pretrained(f'bert-{model_size}-uncased')
    elif model_name == 'roberta':
        if task == 'all':
            model = MTL_Transformer_LSTM_gate(model_name, model_size, args=args)
        else:
            model = RoBERTa(model_size, args=args, num_labels=num_labels)
        tokenizer = RobertaTokenizer.from_pretrained(f'roberta-{model_size}')
    elif model_name == 'bert-gate' and task == 'all':
        model_name = model_name.replace('-gate', '')
        model = GatedModel(model_name, model_size, args=args)
        tokenizer = BertTokenizer.from_pretrained(f'bert-{model_size}-uncased')
    elif model_name == 'roberta-gate' and task == 'all':
        model_name = model_name.replace('-gate', '')
        model = GatedModel(model_name, model_size, args=args)
        tokenizer = RobertaTokenizer.from_pretrained(f'roberta-{model_size}')
        
    # Move model to correct device
    model = model.to(device=device)
    
    
    # prepare data set
    test_file = input("please write the path of test data file：")
    ids, input_ids, mask = read_test_data(tokenizer, test_file)
    
    # load pretrained model
    model_file = input("please write the path of model file：")
    print('your model is: {}'.format(model_file))
    # model_file = './save/models/all_2020-Feb-20_13:33:56.pt'
    saved_model = load(model_file)
    model.load_state_dict(saved_model, strict=False)
    
    print('success load model')
    
    
    
    test_set = TestDataset(ids=ids, input_ids=input_ids, mask=mask)
    test_loader = DataLoader(dataset=test_set, batch_size=bs)
    model.eval()
    
    lines = []
    
    for iteration, (ids, input_ids, length, mask) in enumerate(tqdm(test_loader)):
        ids = ids.to(device=device)
        input_ids = input_ids.to(device=device)
        length = length.to(device=device)
        mask = mask.to(device=device)
        with torch.set_grad_enabled(False):
            all_logits = model(input_ids, length, mask)
            y_pred_A = all_logits[0].argmax(dim=1)
            y_pred_B = all_logits[1].argmax(dim=1)
            y_pred_C = all_logits[2].argmax(dim=1)
        
        ids = ids.tolist()
        y_pred_A = y_pred_A.tolist()
        
        for i in range(len(ids)):
            line = []
            line.append(ids[i])
            line.append('OFF' if y_pred_A[i]==0 else 'NOT')
            lines.append(line)
            
    with open("predictions.csv","w") as csvfile:
        writer = csv.writer(csvfile, lineterminator='\n')
        writer.writerow(['ID','LABEL'])
        writer.writerows(lines)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
