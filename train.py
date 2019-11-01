import os
import copy
import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from data import task_a, task_b, task_c, all_tasks, read_test_file
from config import OLID_PATH, SAVE_PATH
from cli import get_args
from utils import save, get_loss_weight
from datasets import HuggingfaceDataset
from models.bert import BERT, RoBERTa
from transformers import BertTokenizer, RobertaTokenizer

TRAIN_PATH = os.path.join(OLID_PATH, 'olid-training-v1.0.tsv')
datetimestr = datetime.datetime.now().strftime('%Y-%b-%d_%H:%M:%S')

def train_model(model, epochs, dataloaders, criterion, optimizer, scheduler, device, print_iter, patience, task_name):
    # When patience_counter > patience, the training will stop
    patience_counter = 0
    # Statistics to record
    # best_train_acc = 0
    # best_val_acc = 0
    best_train_f1 = 0
    best_val_f1 = 0
    # train_accs = []
    # val_accs = []
    train_losses = []
    val_losses = []
    train_f1 = []
    val_f1 = []
    best_model_weights = None

    for epoch in range(epochs):
        print('\n Epoch {}'.format(epoch))
        print('=' * 20)

        for phase in ['train', 'val']:
            print('Phase [{}]'.format(phase))
            print('-' * 10)

            if phase == 'train':
                model.train() # Set model to training mode
            else:
                model.eval() # Set model to evaluate mode

            dataloader = dataloaders[phase]
            # this_acc = 0
            this_f1 = [0, 0, 0] # macro, micro, weighted
            this_loss = 0
            iter_per_epoch = 0
            for iteration, (inputs, mask, labels) in enumerate(dataloader):
                iter_per_epoch += 1

                inputs = inputs.to(device=device)
                mask = mask.to(device=device)
                labels = labels.to(device=device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    loss, logits = model(inputs, mask, labels)
                    # loss = criterion(logits, labels)
                    y_pred = logits.argmax(dim=1)
                    # acc = torch.sum(y_pred == labels).item() / logits.size(dim=0)
                    this_loss += loss.item()
                    # this_acc += acc
                    this_f1[0] += f1_score(labels.cpu(), y_pred.cpu(), average='macro')
                    this_f1[1] += f1_score(labels.cpu(), y_pred.cpu(), average='micro')
                    this_f1[2] += f1_score(labels.cpu(), y_pred.cpu(), average='weighted')

                    if phase == 'train':
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
                        optimizer.step()
                        # scheduler.step()
                        if iteration % print_iter == 0:
                            print('Iteration {}: loss = {:4f}'.format(iteration, loss))

            this_loss /= iter_per_epoch
            # this_acc /= iter_per_epoch
            this_f1[0] /= iter_per_epoch
            this_f1[1] /= iter_per_epoch

            print('*' * 10)
            print(f'Loss={this_loss}')
            print(f'Macro-F1={this_f1[0]}')
            print(f'Micro-F1={this_f1[1]}')
            print(f'Weighted-F1={this_f1[2]}')
            print('*' * 10)

            if phase == 'train':
                train_losses.append(this_loss)
                # train_accs.append(this_acc)
                train_f1.append(this_f1)
                if this_f1[0] > best_train_f1:
                    best_train_f1 = this_f1[0]
            else:
                patience_counter += 1
                val_losses.append(this_loss)
                # val_accs.append(this_acc)
                val_f1.append(this_f1)
                if this_f1[0] > best_val_f1:
                    best_val_f1 = this_f1[0]
                    patience_counter = 0
                    best_model_weights = copy.deepcopy(model.state_dict())
                elif patience_counter == patience:
                    print('Stop training because running out of patience!')
                    save((
                        train_losses,
                        val_losses,
                        train_f1,
                        val_f1,
                        best_train_f1,
                        best_val_f1
                        # best_train_acc,
                        # best_val_acc,
                        # train_accs,
                        # val_accs
                    ), os.path.join(SAVE_PATH, 'results' + task_name + datetimestr + '.pt'))
                    exit(1)
        print()

    save(best_model_weights, os.path.join(SAVE_PATH, 'best_model_weights' + task_name + datetimestr + '.pt'))
    save((
        train_losses,
        val_losses,
        train_f1,
        val_f1,
        best_train_f1,
        best_val_f1
        # best_train_acc,
        # best_val_acc,
        # train_accs,
        # val_accs
    ), os.path.join(SAVE_PATH, 'results' + task_name + datetimestr + '.pt'))


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
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = args['cuda']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    num_labels = 3 if task == 'c' else 2

    # Set tokenizer for different models
    if model_name == 'bert':
        model = BERT(model_size, num_labels)
        tokenizer = BertTokenizer.from_pretrained(f'bert-{model_size}-uncased')
    elif model_name == 'roberta':
        model = RoBERTa(model_size, num_labels)
        tokenizer = RobertaTokenizer.from_pretrained(f'roberta-{model_size}')

    # Move model to correct device
    model = model.to(device=device)

    # Read in data depends on different subtasks
    data_methods = {'a': task_a, 'b': task_b, 'c': task_c, 'all': all_tasks}
    label_orders = {'a': ['OFF', 'NOT'], 'b': ['TIN', 'UNT'], 'c': ['IND', 'GRP', 'OTH']}
    try:
        ids, token_ids, mask, labels = data_methods[task](TRAIN_PATH, tokenizer=tokenizer, truncate=truncate)
        test_ids, test_token_ids, test_mask, test_labels = read_test_file(task, tokenizer=tokenizer, truncate=truncate)
    except KeyError:
        raise Exception('Incorrect task={}'.format(task))

    cross_entropy_loss_weight = get_loss_weight(np.concatenate((labels, test_labels)), label_orders[task])
    print(f'Label weights: {cross_entropy_loss_weight}')

    dataloaders = {
        'train': DataLoader(
            dataset=HuggingfaceDataset(
                input_ids=token_ids,
                mask=mask,
                labels=labels,
                task=task
            ),
            batch_size=bs,
            shuffle=True
        ),
        'val': DataLoader(
            dataset=HuggingfaceDataset(
                input_ids=test_token_ids,
                mask=test_mask,
                labels=test_labels,
                task=task
            ),
            batch_size=bs
        )
    }

    criterion = torch.nn.CrossEntropyLoss(weight=cross_entropy_loss_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    train_model(
        model=model,
        epochs=epochs,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None,
        device=device,
        print_iter=print_iter,
        patience=patience,
        task_name=task
    )
