import torch
import torch.nn as nn
import torch.optim as optim


class MetaInputDependenceBase:
    """
        Class support computing input-dependence baseline with meta learning
    """
    def __init__(self, actor_critic, num_inner_steps=1):
        self.actor_critic = actor_critic
        self.num_inner_steps = num_inner_steps

    def train_meta_critic(self, rollouts):
        """
            Train the meta critic with rollout experience 
        """
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        # prepare input and output of meta learner
        # All environments share same input job sequences, thus we divide the input to 2 groups
        # each group update its parameters separably and then re-evaluate the new param on other tasks
        # Finally, parameters are updated according to loss computed as re-evaluation phase.
        task_pt = int(num_processes/2)
        task_inputs = rollouts.obs[:, :task_pt, ...]
        task_labels = rollouts.returns[:, task_pt:-1, ...]
        meta_inputs = rollouts.obs[:, task_pt:-1, ...]
        meta_labels = rollouts.returns[:, task_pt:-1, ...]

        for update in range(self.num_inner_steps):

        # train meta network 
        # the actor critic object must be instance of MetaCritic class
        self.actor_critic.train_meta_critic(task_inputs, task_labels, meta_inputs, meta_labels)

    def one_shot_predict_value(self, rollouts):
        """
            Predict the value
        """
