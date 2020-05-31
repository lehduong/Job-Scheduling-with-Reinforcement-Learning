import torch 

from collections import OrderedDict
from torch.optim import Adam
from torch.nn import MSELoss
from .pg import Policy
from .models import ActorMetaCritic


class MetaInputDependentPolicy(Policy):
    def __init__(self, obs_shape, action_shape, base=ActorMetaCritic, base_kwargs=None, num_inner_steps=1, meta_lr=2e-3, meta_criterion=MSELoss):
        super().__init__(obs_shape, action_shape, base, base_kwargs)
        self.num_inner_steps = num_inner_steps
        # TODO: add args for optimizer
        self.lr = meta_lr
        self.criterion = meta_criterion()

    def train_and_predict_meta_critic(self, task_inputs, task_labels, meta_inputs, meta_labels):
        """
            Adapt the meta critic to new input-sequence and predict the values of new observation (with same input sequence)
            :param task_inputs: tuple of (obs, rnn_hxs, masks) - training inputs of roll out with same input sequence
            :param task_labels: array of shape (num_steps, num_envs, 1) - the Monte Carlo approximation of values
            :param meta_inputs: tuple of (obs, rnn_hxs, masks) - testing inputs of roll out with same input sequence as training
            :param meta_labels: array of shape (num_steps, num_envs, 1) - the Monte Carlo approximation of values
            :return: value prediction of meta_inputs and meta gradient of critic
        """
        fast_weights = OrderedDict((name, param) for name, param in self.base.critic.named_parameters())
        task_obs, task_rnn_hxs, task_masks = task_inputs

        # Adapt to new task
        for step in range(self.num_inner_steps):
            if step == 0: 
                task_preds, _, _ = self.base(task_obs, task_rnn_hxs, task_masks)
                task_loss = self.criterion(task_preds, task_labels)
                grads = torch.autograd.grad(task_loss, self.base.critic.parameters(), creat_graph=True)
            else: 
                task_preds, _, _ = self.base(task_obs, task_rnn_hxs, task_masks, fast_weights)
                task_loss = self.criterion(task_preds, task_labels)
                grads = torch.autograd.grad(task_loss, fast_weights.values(), create_graph=True)
            fast_weights = OrderedDict((name, param - self.meta_lr*grad) 
                    for ((name, param), grad) in zip(fast_weights.items(), grads))

        # compute meta grad 
        meta_obs, meta_rnn_hxs, meta_masks = meta_inputs
        meta_preds, _, _ = sel.base(meta_obs, meta_rnn_hxs, meta_masks, fast_weights)
        meta_loss = self.criterion(meta_preds, meta_labels)
        grads = torch.autograd.grad(meta_loss, self.base.critic.parameters())
        meta_grads = {name:g for ((name, _), g) in zip(self.base.critic.named_parameters(), grads)}
        
        return meta_preds, meta_grads

    def get_value(self, inputs, rnn_hxs, masks, state_dict=None):
        value, _, _ = self.base(inputs, rnn_hxs, masks, state_dict)
        return value
    
    def evaluate_actions(self, inputs, rnn_hxs, masks, action, state_dict=None):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks, state_dict)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs
