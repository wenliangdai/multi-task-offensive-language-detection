# Built-in libraries
import copy
import datetime
from typing import Dict, List
# Third-party libraries
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm
# Local files
from utils import save
from config import LABEL_DICT

class Trainer():
    '''
    The trainer for training models.
    It can be used for both single and multi task training.
    Every class function ends with _m is for multi-task training.
    '''
    def __init__(
        self,
        model: nn.Module,
        epochs: int,
        dataloaders: Dict[str, DataLoader],
        criterion: nn.Module,
        loss_weights: List[float],
        clip: bool,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        device: str,
        print_iter: int,
        patience: int,
        task_name: str,
        model_name: str,
        final: bool
    ):
        self.model = model
        self.epochs = epochs
        self.dataloaders = dataloaders
        self.criterion = criterion
        self.loss_weights = loss_weights
        self.clip = clip
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.print_iter = print_iter
        self.patience = patience
        self.task_name = task_name
        self.model_name = model_name
        self.final = final
        self.datetimestr = datetime.datetime.now().strftime('%Y-%b-%d_%H:%M:%S')

        # Evaluation results
        self.train_losses = []
        self.test_losses = []
        self.train_f1 = []
        self.test_f1 = []
        self.best_train_f1 = 0.0
        self.best_test_f1 = 0.0

        # Evaluation results for multi-task
        self.best_train_f1_m = np.array([0, 0, 0], dtype=np.float64)
        self.best_test_f1_m = np.array([0, 0, 0], dtype=np.float64)
        if self.final:
            self.best_train_f1_m = np.array([0, 0, 0, 0], dtype=np.float64)
            self.best_test_f1_m = np.array([0, 0, 0, 0], dtype=np.float64)

    def train(self):
        for epoch in range(self.epochs):
            print(f'Epoch {epoch}')
            print('=' * 20)
            self.train_one_epoch()
            self.test()
            print(f'Best test f1: {self.best_test_f1:.4f}')
            print('=' * 20)

        print('Saving results ...')
        save(
            (self.train_losses, self.test_losses, self.train_f1, self.test_f1, self.best_train_f1, self.best_test_f1),
            f'./save/results/single_{self.task_name}_{self.datetimestr}_{self.best_test_f1:.4f}.pt'
        )

    def train_one_epoch(self):
        self.model.train()
        dataloader = self.dataloaders['train']
        y_pred_all = None
        labels_all = None
        loss = 0
        iters_per_epoch = 0
        for inputs, lens, mask, labels in tqdm(dataloader, desc='Training'):
            iters_per_epoch += 1

            if labels_all is None:
                labels_all = labels.numpy()
            else:
                labels_all = np.concatenate((labels_all, labels.numpy()))

            inputs = inputs.to(device=self.device)
            lens = lens.to(device=self.device)
            mask = mask.to(device=self.device)
            labels = labels.to(device=self.device)

            self.optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                # Forward
                logits = self.model(inputs, lens, mask, labels)
                _loss = self.criterion(logits, labels)
                loss += _loss.item()
                y_pred = logits.argmax(dim=1).cpu().numpy()

                if y_pred_all is None:
                    y_pred_all = y_pred
                else:
                    y_pred_all = np.concatenate((y_pred_all, y_pred))

                # Backward
                _loss.backward()
                if self.clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

        loss /= iters_per_epoch
        f1 = f1_score(labels_all, y_pred_all, average='macro')

        print(f'loss = {loss:.4f}')
        print(f'Macro-F1 = {f1:.4f}')

        self.train_losses.append(loss)
        self.train_f1.append(f1)
        if f1 > self.best_train_f1:
            self.best_train_f1 = f1

    def test(self):
        self.model.eval()
        dataloader = self.dataloaders['test']
        y_pred_all = None
        labels_all = None
        loss = 0
        iters_per_epoch = 0
        for inputs, lens, mask, labels in tqdm(dataloader, desc='Testing'):
            iters_per_epoch += 1

            if labels_all is None:
                labels_all = labels.numpy()
            else:
                labels_all = np.concatenate((labels_all, labels.numpy()))

            inputs = inputs.to(device=self.device)
            lens = lens.to(device=self.device)
            mask = mask.to(device=self.device)
            labels = labels.to(device=self.device)

            with torch.set_grad_enabled(False):
                logits = self.model(inputs, lens, mask, labels)
                _loss = self.criterion(logits, labels)
                y_pred = logits.argmax(dim=1).cpu().numpy()
                loss += _loss.item()

                if y_pred_all is None:
                    y_pred_all = y_pred
                else:
                    y_pred_all = np.concatenate((y_pred_all, y_pred))

        loss /= iters_per_epoch
        f1 = f1_score(labels_all, y_pred_all, average='macro')

        print(f'loss = {loss:.4f}')
        print(f'Macro-F1 = {f1:.4f}')

        self.test_losses.append(loss)
        self.test_f1.append(f1)
        if f1 > self.best_test_f1:
            self.best_test_f1 = f1
            self.save_model()

    def train_m(self):
        for epoch in range(self.epochs):
            print(f'Epoch {epoch}')
            print('=' * 20)
            self.train_one_epoch_m()
            self.test_m()
            print(f'Best test results A: {self.best_test_f1_m[0]:.4f}')
            print(f'Best test results B: {self.best_test_f1_m[1]:.4f}')
            print(f'Best test results C: {self.best_test_f1_m[2]:.4f}')
            if self.final:
                print(f'Best test results Final: {self.best_test_f1_m[3]:.4f}')
            print('=' * 20)

        print('Saving results ...')
        if self.final:
            save(
                (self.train_losses, self.test_losses, self.train_f1, self.test_f1, self.best_train_f1_m, self.best_test_f1_m),
                f'./save/results/mtl_final_{self.datetimestr}_{self.best_test_f1_m[0]:.4f}_{self.best_test_f1_m[3]:.4f}.pt'
            )
        else:
            save(
                (self.train_losses, self.test_losses, self.train_f1, self.test_f1, self.best_train_f1_m, self.best_test_f1_m),
                f'./save/results/mtl_{self.datetimestr}_{self.best_test_f1_m[0]:.4f}.pt'
            )

    def train_one_epoch_m(self):
        self.model.train()
        dataloader = self.dataloaders['train']

        y_pred_all_A = None
        y_pred_all_B = None
        y_pred_all_C = None
        labels_all_A = None
        labels_all_B = None
        labels_all_C = None

        loss = 0
        iters_per_epoch = 0
        for inputs, lens, mask, label_A, label_B, label_C in tqdm(dataloader, desc='Training M'):
            iters_per_epoch += 1

            inputs = inputs.to(device=self.device)
            lens = lens.to(device=self.device)
            mask = mask.to(device=self.device)
            label_A = label_A.to(device=self.device)
            label_B = label_B.to(device=self.device)
            label_C = label_C.to(device=self.device)

            self.optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                # Forward
                # logits_A, logits_B, logits_C = self.model(inputs, mask)
                all_logits = self.model(inputs, lens, mask)
                y_pred_A = all_logits[0].argmax(dim=1).cpu().numpy()
                y_pred_B = all_logits[1][:, 0:2].argmax(dim=1)
                y_pred_C = all_logits[2][:, 0:3].argmax(dim=1)

                Non_null_index_B = label_B != LABEL_DICT['b']['NULL']
                Non_null_label_B = label_B[Non_null_index_B]
                Non_null_pred_B = y_pred_B[Non_null_index_B]

                Non_null_index_C = label_C != LABEL_DICT['c']['NULL']
                Non_null_label_C = label_C[Non_null_index_C]
                Non_null_pred_C = y_pred_C[Non_null_index_C]

                labels_all_A = label_A.numpy() if labels_all_A is None else np.concatenate((labels_all_A, label_A.numpy()))
                labels_all_B = Non_null_label_B.cpu().numpy() if labels_all_B is None else np.concatenate((labels_all_B, Non_null_label_B.cpu().numpy()))
                labels_all_C = Non_null_label_C.cpu().numpy() if labels_all_C is None else np.concatenate((labels_all_C, Non_null_label_C.cpu().numpy()))

                y_pred_all_A = y_pred_A if y_pred_all_A is None else np.concatenate((y_pred_all_A, y_pred_A))
                y_pred_all_B = Non_null_pred_B.cpu().numpy() if y_pred_all_B is None else np.concatenate((y_pred_all_B, Non_null_pred_B.cpu().numpy()))
                y_pred_all_C = Non_null_pred_C.cpu().numpy() if y_pred_all_C is None else np.concatenate((y_pred_all_C, Non_null_pred_C.cpu().numpy()))

                # f1[0] += self.calc_f1(label_A, y_pred_A)
                # f1[1] += self.calc_f1(Non_null_label_B, Non_null_pred_B)
                # f1[2] += self.calc_f1(Non_null_label_C, Non_null_pred_C)

                _loss = self.loss_weights[0] * self.criterion(all_logits[0], label_A)
                _loss += self.loss_weights[1] * self.criterion(all_logits[1], label_B)
                _loss += self.loss_weights[2] * self.criterion(all_logits[2], label_C)

                if self.final:
                    y_pred_final = all_logits[3].argmax(dim=1)
                    _loss += self.loss_weights[3] * self.criterion(all_logits[3], label_A)
                    f1[3] += self.calc_f1(label_A, y_pred_final)

                loss += _loss.item()

                # Backward
                _loss.backward()
                if self.clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

        loss /= iters_per_epoch
        f1_A = f1_score(labels_all_A, y_pred_all_A, average='macro')
        f1_B = f1_score(labels_all_B, y_pred_all_B, average='macro')
        f1_C = f1_score(labels_all_C, y_pred_all_C, average='macro')

        print(f'loss = {loss:.4f}')
        print(f'A: {f1_A:.4f}')
        print(f'B: {f1_B:.4f}')
        print(f'C: {f1_C:.4f}')
        if self.final:
            print(f'Final: {f1[3][0]:.4f}, {f1[3][1]:.4f}, {f1[3][2]:.4f}')

        self.train_losses.append(loss)
        self.train_f1.append([f1_A, f1_B, f1_C])

        if f1_A > self.best_train_f1_m[0]:
            self.best_train_f1_m[0] = f1_A
        if f1_B > self.best_train_f1_m[1]:
            self.best_train_f1_m[1] = f1_B
        if f1_C > self.best_train_f1_m[2]:
            self.best_train_f1_m[2] = f1_C

        # for i in range(len(f1)):
        #     for j in range(len(f1[0])):
        #         if f1[i][j] > self.best_train_f1_m[i][j]:
        #             self.best_train_f1_m[i][j] = f1[i][j]
        #             if not self.final and i == 0 and j == 0:
        #                 self.save_model()
        #             if self.final and i == 3 and j == 0:
        #                 self.save_model()

    def test_m(self):
        self.model.eval()
        dataloader = self.dataloaders['test']
        if self.final:
            f1 = np.concatenate((f1, [[0, 0, 0]]))
        loss = 0
        iters_per_epoch = 0

        y_pred_all_A = None
        y_pred_all_B = None
        y_pred_all_C = None
        labels_all_A = None
        labels_all_B = None
        labels_all_C = None

        for inputs, lens, mask, label_A, label_B, label_C in tqdm(dataloader, desc='Test M'):
            iters_per_epoch += 1

            labels_all_A = label_A.numpy() if labels_all_A is None else np.concatenate((labels_all_A, label_A.numpy()))
            labels_all_B = label_B.numpy() if labels_all_B is None else np.concatenate((labels_all_B, label_B.numpy()))
            labels_all_C = label_C.numpy() if labels_all_C is None else np.concatenate((labels_all_C, label_C.numpy()))

            inputs = inputs.to(device=self.device)
            lens = lens.to(device=self.device)
            mask = mask.to(device=self.device)
            label_A = label_A.to(device=self.device)
            label_B = label_B.to(device=self.device)
            label_C = label_C.to(device=self.device)

            with torch.set_grad_enabled(False):
                all_logits = self.model(inputs, lens, mask)
                y_pred_A = all_logits[0].argmax(dim=1).cpu().numpy()
                y_pred_B = all_logits[1].argmax(dim=1).cpu().numpy()
                y_pred_C = all_logits[2].argmax(dim=1).cpu().numpy()

                # f1[0] += self.calc_f1(label_A, y_pred_A)
                # f1[1] += self.calc_f1(label_B, y_pred_B)
                # f1[2] += self.calc_f1(label_C, y_pred_C)

                y_pred_all_A = y_pred_A if y_pred_all_A is None else np.concatenate((y_pred_all_A, y_pred_A))
                y_pred_all_B = y_pred_B if y_pred_all_B is None else np.concatenate((y_pred_all_B, y_pred_B))
                y_pred_all_C = y_pred_C if y_pred_all_C is None else np.concatenate((y_pred_all_C, y_pred_C))

                _loss = self.loss_weights[0] * self.criterion(all_logits[0], label_A)
                _loss += self.loss_weights[1] * self.criterion(all_logits[1], label_B)
                _loss += self.loss_weights[2] * self.criterion(all_logits[2], label_C)

                if self.final:
                    y_pred_final = all_logits[3].argmax(dim=1)
                    _loss += self.loss_weights[3] * self.criterion(all_logits[3], label_A)
                    f1[3] += self.calc_f1(label_A, y_pred_final)

                loss += _loss.item()

        loss /= iters_per_epoch
        f1_A = f1_score(labels_all_A, y_pred_all_A, average='macro')
        f1_B = f1_score(labels_all_B, y_pred_all_B, average='macro')
        f1_C = f1_score(labels_all_C, y_pred_all_C, average='macro')

        print(f'loss = {loss:.4f}')
        print(f'A: {f1_A:.4f}')
        print(f'B: {f1_B:.4f}')
        print(f'C: {f1_C:.4f}')

        if self.final:
            print(f'Final: {f1[3][0]:.4f}, {f1[3][1]:.4f}, {f1[3][2]:.4f}')

        self.test_losses.append(loss)
        self.test_f1.append([f1_A, f1_B, f1_C])

        if f1_A > self.best_test_f1_m[0]:
            self.best_test_f1_m[0] = f1_A
            self.save_model()
        if f1_B > self.best_test_f1_m[1]:
            self.best_test_f1_m[1] = f1_B
        if f1_C > self.best_test_f1_m[2]:
            self.best_test_f1_m[2] = f1_C

        # for i in range(len(f1)):
        #     for j in range(len(f1[0])):
        #         if f1[i][j] > self.best_test_f1_m[i][j]:
        #             self.best_test_f1_m[i][j] = f1[i][j]
        #             if i == 0 and j == 0:
        #                 self.save_model()

    def calc_f1(self, labels, y_pred):
        return np.array([
            f1_score(labels.cpu(), y_pred.cpu(), average='macro'),
            f1_score(labels.cpu(), y_pred.cpu(), average='micro'),
            f1_score(labels.cpu(), y_pred.cpu(), average='weighted')
        ], np.float64)

    def printing(self, loss, f1):
        print(f'loss = {loss:.4f}')
        print(f'Macro-F1 = {f1[0]:.4f}')
        # print(f'Micro-F1 = {f1[1]:.4f}')
        # print(f'Weighted-F1 = {f1[2]:.4f}')

    def save_model(self):
        print('Saving model...')
        if self.task_name == 'all':
            filename = f'./save/models/{self.task_name}_{self.model_name}_{self.best_test_f1_m[0]}.pt'
        else:
            filename = f'./save/models/{self.task_name}_{self.model_name}_{self.best_test_f1}.pt'
        save(copy.deepcopy(self.model.state_dict()), filename)
