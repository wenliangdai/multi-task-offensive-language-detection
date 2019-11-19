# Built-in libraries
import copy
import datetime
from typing import Dict, List
# Third-party libraries
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
# Local files
from utils import save

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
        model_name: str
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

        # Evaluation results
        self.train_losses = []
        self.test_losses = []
        self.train_f1 = []
        self.test_f1 = []
        self.best_train_f1 = np.array([0, 0, 0], dtype=np.float64)
        self.best_test_f1 = np.array([0, 0, 0], dtype=np.float64)

        # Evaluation results for multi-task
        self.best_train_f1_m = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ], dtype=np.float64)
        self.best_test_f1_m = np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ], dtype=np.float64)

    def train(self):
        for epoch in range(self.epochs):
            print(f'Epoch {epoch}')
            print('=' * 20)
            print('Training...')
            self.train_one_epoch()
            print('Testing...')
            self.test()
            print(f'Best test results: {self.best_test_f1[0]:.4f}, {self.best_test_f1[1]:.4f}, {self.best_test_f1[2]:.4f}')
            print('=' * 20)

    def train_one_epoch(self):
        self.model.train()
        dataloader = self.dataloaders['train']
        f1 = np.array([0, 0, 0], dtype=np.float64) # [macro, micro, weighted]
        loss = 0
        iters_per_epoch = 0
        for iteration, (inputs, mask, labels) in enumerate(dataloader):
            iters_per_epoch += 1

            inputs = inputs.to(device=self.device)
            mask = mask.to(device=self.device)
            labels = labels.to(device=self.device)

            self.optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                # Forward
                logits = self.model(inputs, mask, labels)
                _loss = self.criterion(logits, labels)
                y_pred = logits.argmax(dim=1)
                loss += _loss.item()
                f1 += self.calc_f1(labels, y_pred)
                # Backward
                _loss.backward()
                if self.clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                if iteration % self.print_iter == 0:
                    print(f'Iteration {iteration}: loss = {_loss:.4f}')

        loss /= iters_per_epoch
        f1 /= iters_per_epoch

        self.printing(loss, f1)

        self.train_losses.append(loss)
        self.train_f1.append(f1)
        for i in range(len(f1)):
            if f1[i] > self.best_train_f1[i]:
                self.best_train_f1[i] = f1[i]

    def test(self):
        self.model.eval()
        dataloader = self.dataloaders['test']
        f1 = np.array([0, 0, 0], np.float64) # [macro, micro, weighted]
        loss = 0
        iters_per_epoch = 0
        for iteration, (inputs, mask, labels) in enumerate(dataloader):
            iters_per_epoch += 1

            inputs = inputs.to(device=self.device)
            mask = mask.to(device=self.device)
            labels = labels.to(device=self.device)

            with torch.set_grad_enabled(False):
                logits = self.model(inputs, mask, labels)
                _loss = self.criterion(logits, labels)
                y_pred = logits.argmax(dim=1)
                loss += _loss.item()
                f1 += self.calc_f1(labels, y_pred)

        loss /= iters_per_epoch
        f1 /= iters_per_epoch

        self.printing(loss, f1)

        self.test_losses.append(loss)
        self.test_f1.append(f1)
        for i in range(len(f1)):
            if f1[i] > self.best_test_f1[i]:
                self.best_test_f1[i] = f1[i]

    def train_m(self):
        for epoch in range(self.epochs):
            print(f'Epoch {epoch}')
            print('=' * 20)
            print('Training...')
            self.train_one_epoch_m()
            print('Testing...')
            self.test_m()
            print(f'Best test results A: {self.best_test_f1_m[0][0]:.4f}, {self.best_test_f1_m[0][1]:.4f}, {self.best_test_f1_m[0][2]:.4f}')
            print(f'Best test results B: {self.best_test_f1_m[1][0]:.4f}, {self.best_test_f1_m[1][1]:.4f}, {self.best_test_f1_m[1][2]:.4f}')
            print(f'Best test results C: {self.best_test_f1_m[2][0]:.4f}, {self.best_test_f1_m[2][1]:.4f}, {self.best_test_f1_m[2][2]:.4f}')
            print('=' * 20)

    def train_one_epoch_m(self):
        self.model.train()
        dataloader = self.dataloaders['train']
        f1 = np.array([
            [0, 0, 0], # [macro, micro, weighted]
            [0, 0, 0],
            [0, 0, 0]
        ], dtype=np.float64)
        loss = 0
        iters_per_epoch = 0
        for iteration, (inputs, mask, label_A, label_B, label_C) in enumerate(dataloader):
            iters_per_epoch += 1

            inputs = inputs.to(device=self.device)
            mask = mask.to(device=self.device)
            label_A = label_A.to(device=self.device)
            label_B = label_B.to(device=self.device)
            label_C = label_C.to(device=self.device)

            self.optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                # Forward
                logits_A, logits_B, logits_C = self.model(inputs, mask)
                y_pred_A = logits_A.argmax(dim=1)
                y_pred_B = logits_B.argmax(dim=1)
                y_pred_C = logits_C.argmax(dim=1)

                _loss = self.loss_weights[0] * self.criterion(logits_A, label_A)
                _loss += self.loss_weights[1] * self.criterion(logits_B, label_B)
                _loss += self.loss_weights[2] * self.criterion(logits_C, label_C)

                loss += _loss.item()
                f1[0] += self.calc_f1(label_A, y_pred_A)
                f1[1] += self.calc_f1(label_B, y_pred_B)
                f1[2] += self.calc_f1(label_C, y_pred_C)
                # Backward
                _loss.backward()
                if self.clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
                if iteration % self.print_iter == 0:
                    print(f'Iteration {iteration}: loss = {_loss:.4f}')

        loss /= iters_per_epoch
        f1 /= iters_per_epoch

        print(f'loss = {loss:.4f}')
        print(f'A: {f1[0][0]:.4f}, {f1[0][1]:.4f}, {f1[0][2]:.4f}')
        print(f'B: {f1[1][0]:.4f}, {f1[1][1]:.4f}, {f1[1][2]:.4f}')
        print(f'C: {f1[2][0]:.4f}, {f1[2][1]:.4f}, {f1[2][2]:.4f}')

        self.train_losses.append(loss)
        self.train_f1.append(f1)
        for i in range(len(f1)):
            for j in range(len(f1[0])):
                if f1[i][j] > self.best_train_f1_m[i][j]:
                    self.best_train_f1_m[i][j] = f1[i][j]

    def test_m(self):
        self.model.eval()
        dataloader = self.dataloaders['test']
        f1 = np.array([
            [0, 0, 0], # [macro, micro, weighted]
            [0, 0, 0],
            [0, 0, 0]
        ], dtype=np.float64)
        loss = 0
        iters_per_epoch = 0
        for iteration, (inputs, mask, label_A, label_B, label_C) in enumerate(dataloader):
            iters_per_epoch += 1

            inputs = inputs.to(device=self.device)
            mask = mask.to(device=self.device)
            label_A = label_A.to(device=self.device)
            label_B = label_B.to(device=self.device)
            label_C = label_C.to(device=self.device)

            with torch.set_grad_enabled(False):
                logits_A, logits_B, logits_C = self.model(inputs, mask)
                y_pred_A = logits_A.argmax(dim=1)
                y_pred_B = logits_B.argmax(dim=1)
                y_pred_C = logits_C.argmax(dim=1)

                _loss = (self.criterion(logits_A, label_A) +
                         self.criterion(logits_B, label_B) +
                         self.criterion(logits_C, label_C))
                loss += _loss.item()
                f1[0] += self.calc_f1(label_A, y_pred_A)
                f1[1] += self.calc_f1(label_B, y_pred_B)
                f1[2] += self.calc_f1(label_C, y_pred_C)

        loss /= iters_per_epoch
        f1 /= iters_per_epoch

        print(f'loss = {loss:.4f}')
        print(f'A: {f1[0][0]:.4f}, {f1[0][1]:.4f}, {f1[0][2]:.4f}')
        print(f'B: {f1[1][0]:.4f}, {f1[1][1]:.4f}, {f1[1][2]:.4f}')
        print(f'C: {f1[2][0]:.4f}, {f1[2][1]:.4f}, {f1[2][2]:.4f}')

        self.test_losses.append(loss)
        self.test_f1.append(f1)
        for i in range(len(f1)):
            for j in range(len(f1[0])):
                if f1[i][j] > self.best_test_f1_m[i][j]:
                    self.best_test_f1_m[i][j] = f1[i][j]

    def calc_f1(self, labels, y_pred):
        return np.array([
            f1_score(labels.cpu(), y_pred.cpu(), average='macro'),
            f1_score(labels.cpu(), y_pred.cpu(), average='micro'),
            f1_score(labels.cpu(), y_pred.cpu(), average='weighted')
        ], np.float64)

    def printing(self, loss, f1):
        print(f'loss = {loss:.4f}')
        print(f'Macro-F1 = {f1[0]:.4f}')
        print(f'Micro-F1 = {f1[1]:.4f}')
        print(f'Weighted-F1 = {f1[2]:.4f}')

    def save_model(self):
        datetimestr = datetime.datetime.now().strftime('%Y-%b-%d_%H:%M:%S')
        save(
            copy.deepcopy(self.model.state_dict()),
            f'./save/model_weights_{self.task_name}_{self.model_name}_{datetimestr}.pt'
        )
        save((
            self.train_losses,
            self.test_losses,
            self.train_f1,
            self.test_f1,
            self.best_train_f1,
            self.best_test_f1
        ), f'./save/results_{self.task_name}_{self.model_name}_{datetimestr}.pt')
