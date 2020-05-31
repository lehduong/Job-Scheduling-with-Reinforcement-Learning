import copy
import torch 

from collections import OrderedDict
from torch.optim import Adam, SGD
from torch.nn import MSELoss
from .pg import Policy


class MetaInputDependentPolicy(Policy):
    def __init__(self, obs_shape, action_shape, base=None, base_kwargs=None, num_inner_steps=1, meta_lr=2e-3, meta_criterion=MSELoss):
        super().__init__(obs_shape, action_shape, base, base_kwargs)
        self.num_inner_steps = num_inner_steps
        # TODO: add args for optimizer
        self.lr = meta_lr
        self.meta_criterion = meta_criterion()

    @property
    def criterion(self):
        return self.meta_criterion 

    def train_and_predict_meta_critic(self, task_inputs, task_labels, meta_inputs, meta_labels):
        """
            Adapt the meta critic to new input-sequence and predict the values of new observation (with same input sequence)
                For simplicity, we adopt the **First-order MAML** https://arxiv.org/abs/1803.02999 \

            :param task_inputs: tuple of (obs, rnn_hxs, masks) - training inputs of roll out with same input sequence
            :param task_labels: array of shape (num_steps, num_envs, 1) - the Monte Carlo approximation of values
            :param meta_inputs: tuple of (obs, rnn_hxs, masks) - testing inputs of roll out with same input sequence as training
            :param meta_labels: array of shape (num_steps, num_envs, 1) - the Monte Carlo approximation of values
            :return: value prediction of meta_inputs and meta gradient of critic
        """
        # create new net and exclusively update this network
        fast_net = copy.deepcopy(self.base)
        task_optimizer = SGD(fast_net.critic.parameters(), lr=self.lr)

        task_obs, task_rnn_hxs, task_masks = task_inputs

        # Adapt to new task
        for step in range(self.num_inner_steps):
            task_preds, _, _ = fast_net(task_obs, task_rnn_hxs, task_masks)
            task_loss = self.meta_criterion(task_preds, task_labels)

            # update the fast-adapted network
            task_optimizer.zero_grad()
            task_loss.backward()
            task_optimizer.step()

        # compute meta grad 
        meta_obs, meta_rnn_hxs, meta_masks = meta_inputs
        meta_preds, _, _ = fast_net(meta_obs, meta_rnn_hxs, meta_masks)
        meta_loss = self.meta_criterion(meta_preds, meta_labels)
        grads = torch.autograd.grad(meta_loss, fast_net.critic.parameters())
        meta_grads = {name:g for ((name, _), g) in zip(fast_net.critic.named_parameters(), grads)}
        
        return meta_preds, meta_grads

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value
    
    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs
