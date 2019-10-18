import os
import copy
import datetime
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from data import bert_all_tasks, bert_task_a, bert_task_b, bert_task_c, read_test_file
from config import OLID_PATH, SAVE_PATH
from cli import get_args
from utils import save
from datasets import BERTDataset
from models.bert import BERT_BASE

TRAIN_PATH = os.path.join(OLID_PATH, 'olid-training-v1.0.tsv')
datetimestr = datetime.datetime.now().strftime('%Y-%b-%d_%H:%M:%S')

def train_model(model, epochs, dataloaders, criterion, optimizer, scheduler, device, print_iter, patience, task_name):
    # When patience_counter > patience, the training will stop
    patience_counter = 0
    # Statistics to record
    best_train_acc = 0
    best_val_acc = 0
    best_train_f1 = 0
    best_val_f1 = 0
    train_accs = []
    val_accs = []
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
            this_acc = 0
            this_f1 = [0, 0]
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
                    acc = torch.sum(y_pred == labels).item() / logits.size(dim=0)
                    this_loss += loss.item()
                    this_acc += acc
                    this_f1[0] += f1_score(labels.cpu(), y_pred.cpu(), average='macro')
                    this_f1[1] += f1_score(labels.cpu(), y_pred.cpu(), average='micro')

                    if phase == 'train':
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
                        optimizer.step()
                        # scheduler.step()
                        if iteration % print_iter == 0:
                            print('Iteration {}: loss = {:4f}'.format(iteration, loss))

            this_loss /= iter_per_epoch
            this_acc /= iter_per_epoch
            this_f1[0] /= iter_per_epoch
            this_f1[1] /= iter_per_epoch
            print('[Loss = {:4f}, Acc = {:4f}, Macro-F1 = {:4f}, Micro-F1 = {:4f}]'.format(this_loss, this_acc, this_f1[0], this_f1[1]))

            if phase == 'train':
                train_losses.append(this_loss)
                train_accs.append(this_acc)
                train_f1.append(this_f1)
                if this_f1[0] > best_train_f1:
                    best_train_f1 = this_f1[0]
            else:
                patience_counter += 1
                val_losses.append(this_loss)
                val_accs.append(this_acc)
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
                        best_train_acc,
                        best_val_acc,
                        train_accs,
                        val_accs
                    ), os.path.join(SAVE_PATH, 'results' + task_name + datetimestr + '.pt'))
                    exit(1)
        print()

    save(best_model_weights, os.path.join(SAVE_PATH, 'best_model_weights' + task_name + datetimestr + '.pt'))
    save((
        train_losses,
        val_losses,
        best_train_acc,
        best_val_acc,
        train_accs,
        val_accs
    ), os.path.join(SAVE_PATH, 'results' + task_name + datetimestr + '.pt'))


if __name__ == '__main__':
    # Get command line arguments
    args = get_args()
    task = args['task']

    # Fix seed for reproducibility
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Set device
    os.environ["CUDA_VISIBLE_DEVICES"] = args['cuda']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    cross_entropy_loss_weight = None
    test_ids, test_token_ids, test_mask, test_labels = read_test_file(task, truncate=args['truncate'])
    if task == 'a':
        ids, token_ids, mask, labels = bert_task_a(TRAIN_PATH, truncate=args['truncate'])
        nums_off = np.sum(labels == 'OFF') + np.sum(test_labels == 'OFF')
        nums_not = np.sum(labels == 'NOT') + np.sum(test_labels == 'NOT')
        total = nums_off + nums_not
        cross_entropy_loss_weight = torch.tensor([nums_off / total, nums_not / total])
    elif task == 'b':
        ids, token_ids, mask, labels = bert_task_b(TRAIN_PATH, truncate=args['truncate'])
        nums_tin = np.sum(labels == 'TIN') + np.sum(test_labels == 'TIN')
        nums_unt = np.sum(labels == 'UNT') + np.sum(test_labels == 'UNT')
        total = nums_tin + nums_unt
        cross_entropy_loss_weight = torch.tensor([nums_tin / total, nums_unt / total])
    elif task == 'c':
        ids, token_ids, mask, labels = bert_task_c(TRAIN_PATH, truncate=args['truncate'])
        nums_ind = np.sum(labels == 'IND') + np.sum(test_labels == 'IND')
        nums_grp = np.sum(labels == 'GRP') + np.sum(test_labels == 'GRP')
        nums_oth = np.sum(labels == 'OTH') + np.sum(test_labels == 'OTH')
        total = nums_ind + nums_grp + nums_oth
        cross_entropy_loss_weight = torch.tensor([nums_ind / total, nums_grp / total, nums_oth / total])
    else:
        raise Exception('Incorrect task={}'.format(task))

    dataloaders = {
        'train': DataLoader(
            dataset=BERTDataset(
                input_ids=token_ids,
                mask=mask,
                labels=labels,
                task=task
            ),
            batch_size=args['batch_size'],
            shuffle=True
        ),
        'val': DataLoader(
            dataset=BERTDataset(
                input_ids=test_token_ids,
                mask=test_mask,
                labels=test_labels,
                task=task
            ),
            batch_size=args['batch_size']
        )
    }

    model = BERT_BASE()
    model = model.to(device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=cross_entropy_loss_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])

    train_model(
        model=model,
        epochs=args['epochs'],
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None,
        device=device,
        print_iter=args['print_iter'],
        patience=args['patience'],
        task_name=args['task']
    )
