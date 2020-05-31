import torch
import torch.nn as nn
import torch.optim as optim

from core.algorithms.a2c_acktr import A2C_ACKTR

class MetaInputDependentA2C(A2C_ACKTR):
    """
        Class support computing input-dependence baseline with meta learning
    """
    def __init__(self,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 alpha=None,
                 max_grad_norm=None,
                 acktr=False,
                 num_inner_steps=1):
        super().__init__(actor_critic, value_loss_coef, entropy_coef, lr, eps, alpha, max_grad_norm, acktr)
        self.num_inner_steps = num_inner_steps

    def adapt_and_predict(self, rollouts):
        """
            Train the meta critic with rollout experience and return the predicted values
            
            The adapted algorithm is described in Algorithm 1 of the paper https://arxiv.org/abs/1807.02264 \
                    we split the rollout into 2 half, the critic parameters adapting to the first half will give 
                    the prediction for second half. The critic parameters adapting to the second half will give
                    the prediction for first half.
            :param rollouts: RolloutStorage's instance
            :return: input-dependent values 
        """
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        # prepare input and output of meta learner
        # ie splitting them into 2
        task_pt = int(num_processes/2)
        # first half
        first_obs = rollouts.obs[:-1, :task_pt, ...].view(-1, *obs_shape)  # num_steps * num_processes * input_shape
        first_rnn_hxs = rollouts.recurrent_hidden_states[0, :task_pt].view(-1, self.actor_critic.recurrent_hidden_state_size)
        first_mask = rollouts.masks[:-1, :task_pt].view(-1, 1)
        first_inputs = (first_obs, first_rnn_hxs, first_mask)
        first_labels = rollouts.returns[:-1, :task_pt, ...].view(-1, 1)
        # second half
        second_obs = rollouts.obs[:-1, task_pt:, ...].view(-1, *obs_shape)  # num_steps * num_processes * input_shape
        second_rnn_hxs = rollouts.recurrent_hidden_states[0, task_pt:].view(-1, self.actor_critic.recurrent_hidden_state_size)
        second_mask = rollouts.masks[:-1, task_pt:].view(-1, 1)
        second_inputs = (second_obs, second_rnn_hxs, second_mask)
        second_labels = rollouts.returns[:-1, task_pt:, ...].view(-1, 1)

        # train meta network 
        # the actor critic object must be instance of MetaCritic class
        second_value, second_meta_grads = self.actor_critic.train_and_pred_meta_critic(first_inputs, first_labels, second_inputs, second_labels)
        first_value, first_meta_grads = self.actor_critic.train_and_pred_meta_critic(second_inputs, second_labels, first_inputs, first_labels)
        values = np.vstack([first_value, second_value])
        self.update_meta_grads([first_meta_grads, second_meta_grads])

        return values

    def update_meta_grads(self, grads):
        """
            Cummulate gradients across tasks 
            :param grads: list of OrderedDict - each element is the gradient from a task
        """
        keys = grads[0].keys()
        gradients = {k: sum(grad[k] for grad in grads) for k in keys}

        hooks = []
        for (k,v) in self.actor_critic.base.critic.named_parameters():
            def get_closure():
                key = k
                def replace_grad(grad):
                    return gradients[key]
                return replace_grad
            hooks.append(v.register_hook(get_closure()))
        # compute grad for curr step 
        self.meta_optimizer.step()
        self.meta_optimizer.zero_grad()

        for h in hooks:
            h.remove()

