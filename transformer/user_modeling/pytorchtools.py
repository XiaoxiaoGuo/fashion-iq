import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, args=None, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = -np.Inf
        self.delta = delta
        # self.save_path = args.save_model + '.chkpt'
        self.args = args
        self.epoch_i = 0

    def __call__(self, val_loss, model, epoch_i):

        score = val_loss
        self.epoch_i = epoch_i

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch_i)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch_i)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch_i):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation score changed ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'settings': self.args,
            'epoch': epoch_i}
        torch.save(checkpoint, self.args.save_model + '.chkpt')
        self.val_loss_min = val_loss