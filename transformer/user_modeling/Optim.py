'''A wrapper class for optimizer '''
import numpy as np
import torch
import torch.optim
# class ScheduledOptim():
#     '''A simple wrapper class for learning rate scheduling'''

#     def __init__(self, optimizer, d_model, n_warmup_steps):
#         self._optimizer = optimizer
#         self.n_warmup_steps = n_warmup_steps
#         self.n_current_steps = 0
#         self.init_lr = np.power(d_model, -0.5)

#     def step_and_update_lr(self):
#         "Step with the inner optimizer"
#         self._update_learning_rate()
#         self._optimizer.step()

#     def zero_grad(self):
#         "Zero out the gradients by the inner optimizer"
#         self._optimizer.zero_grad()

#     def _get_lr_scale(self):
#         return np.min([
#             np.power(self.n_current_steps, -0.5),
#             np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

#     def _update_learning_rate(self):
#         ''' Learning rate scheduling per step '''

#         self.n_current_steps += 1
#         lr = self.init_lr * self._get_lr_scale()

#         for param_group in self._optimizer.param_groups:
#             param_group['lr'] = lr

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
def get_std_opt(model,opt):
    return NoamOpt(opt.d_model, 2, opt.n_warmup_steps,
            torch.optim.Adam(model.get_trainable_parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

