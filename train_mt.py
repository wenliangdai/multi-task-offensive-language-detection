import os
# import copy
import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from data import all_tasks, read_test_file_all
from config import OLID_PATH, SAVE_PATH
from cli import get_args
from utils import save, get_loss_weight
from datasets import HuggingfaceMTDataset
from models.bert import MTModel
from transformers import BertTokenizer, RobertaTokenizer
from typing import Dict, Any

TRAIN_PATH = os.path.join(OLID_PATH, 'olid-training-v1.0.tsv')
datetimestr = datetime.datetime.now().strftime('%Y-%b-%d_%H:%M:%S')

def train_model(
    model: Any,
    epochs: int,
    dataloaders: Dict[str, DataLoader],
    criterions: Any,
    optimizer: Any,
    scheduler: Any,
    device: Any,
    print_iter: int,
    patience: int,
    task_name: str,
    model_name: str
):
    # When patience_counter > patience, the training will stop
    patience_counter = 0
    # Statistics to record
    # best_train_f1 = 0
    # best_val_f1 = 0
    train_losses = []
    val_losses = []
    train_f1 = []
    val_f1 = []
    # best_model_weights = None

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
            this_f1 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64)
            this_loss = 0
            iter_per_epoch = 0
            for iteration, (inputs, mask, labels) in enumerate(dataloader):
                iter_per_epoch += 1

                inputs = inputs.to(device=device)
                mask = mask.to(device=device)
                labels = [l.to(device=device) for l in labels]

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    logits_list = model(inputs, mask)
                    losses = np.array([criterions[i](logits_list[i], labels[i]) for i in range(3)])
                    loss_weights = np.array([1.0, 1.0, 1.0]) / 3
                    losses = losses * loss_weights
                    loss = losses[0] + losses[1] + losses[2]
                    y_preds = [logits_list[i].argmax(dim=1).cpu() for i in range(3)]
                    this_loss += np.sum([loss.item() for loss in losses])

                    for i in range(3):
                        this_f1[i] += eval(labels[i].cpu(), y_preds[i])

                    if phase == 'train':
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
                        optimizer.step()
                        # scheduler.step()
                        if iteration % print_iter == 0:
                            print(f'Iteration {iteration}: loss = {loss:5f}')

            this_loss /= iter_per_epoch
            this_f1 /= iter_per_epoch

            print('*' * 10)
            print(f'Loss        = {this_loss:.5f}')
            print(f'Macro-F1    = {this_f1[0][0]:.5f}\t{this_f1[1][0]:.5f}\t{this_f1[2][0]:.5f}')
            print(f'Micro-F1    = {this_f1[0][1]:.5f}\t{this_f1[1][1]:.5f}\t{this_f1[2][1]:.5f}')
            print(f'Weighted-F1 = {this_f1[0][2]:.5f}\t{this_f1[1][2]:.5f}\t{this_f1[2][2]:.5f}')
            print('*' * 10)

            if phase == 'train':
                train_losses.append(this_loss)
                train_f1.append(this_f1)
                # if this_f1[0] > best_train_f1:
                #     best_train_f1 = this_f1[0]
            else:
                patience_counter += 1
                val_losses.append(this_loss)
                val_f1.append(this_f1)

                # if this_f1[0] > best_val_f1:
                #     best_val_f1 = this_f1[0]
                #     patience_counter = 0
                #     best_model_weights = copy.deepcopy(model.state_dict())
                # elif patience_counter == patience:
                #     print('Stop training because running out of patience!')
                #     save((
                #         train_losses,
                #         val_losses,
                #         train_f1,
                #         val_f1,
                #         best_train_f1,
                #         best_val_f1
                #     ), f'{SAVE_PATH}/results_{task_name}_{model_name}_{datetimestr}.pt')
                #     exit(1)
        print()

    # save(best_model_weights, f'{SAVE_PATH}/best_model_weights_{task_name}_{model_name}_{datetimestr}.pt')
    save((
        train_losses,
        val_losses,
        train_f1,
        val_f1,
        # best_train_f1,
        # best_val_f1
    ), f'{SAVE_PATH}/results_{task_name}_{model_name}_{datetimestr}.pt')


def eval(y, y_pred):
    f1s = []
    f1s.append(f1_score(y.cpu(), y_pred.cpu(), average='macro'))
    f1s.append(f1_score(y.cpu(), y_pred.cpu(), average='micro'))
    f1s.append(f1_score(y.cpu(), y_pred.cpu(), average='weighted'))
    return np.array(f1s)


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

    # Move model to correct device
    model = MTModel(model_name, model_size)
    model = model.to(device=device)

    # Set tokenizer for different models
    if model_name == 'bert':
        tokenizer = BertTokenizer.from_pretrained(f'bert-{model_size}-uncased')
    elif model_name == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(f'roberta-{model_size}')

    # Read in data depends on different subtasks
    ids, token_ids, mask, label_a, label_b, label_c = all_tasks(TRAIN_PATH, tokenizer=tokenizer, truncate=truncate)
    test_ids, test_token_ids, test_mask, test_label_a, test_label_b, test_label_c = read_test_file_all(tokenizer)
    labels = {'a': label_a, 'b': label_b, 'c': label_c}
    test_labels = {'a': test_label_a, 'b': test_label_b, 'c': test_label_c}
    label_orders = {'a': ['OFF', 'NOT'], 'b': ['TIN', 'UNT', 'NULL'], 'c': ['IND', 'GRP', 'OTH', 'NULL']}

    cross_entropy_loss_weights = [get_loss_weight(np.concatenate((labels[t], test_labels[t])), label_orders[t]) for t in ['a', 'b', 'c']]
    # print(f'Label weights: {cross_entropy_loss_weight}')

    dataloaders = {
        'train': DataLoader(
            dataset=HuggingfaceMTDataset(
                input_ids=token_ids,
                mask=mask,
                labels=labels
            ),
            batch_size=bs,
            shuffle=True
        ),
        'val': DataLoader(
            dataset=HuggingfaceMTDataset(
                input_ids=test_token_ids,
                mask=test_mask,
                labels=test_labels
            ),
            batch_size=bs
        )
    }

    criterions = [torch.nn.CrossEntropyLoss(weight=w.to(device=device)) for w in cross_entropy_loss_weights]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    train_model(
        model=model,
        epochs=epochs,
        dataloaders=dataloaders,
        criterions=criterions,
        optimizer=optimizer,
        scheduler=None,
        device=device,
        print_iter=print_iter,
        patience=patience,
        task_name=task,
        model_name=model_name
    )
