import copy

from torch import nn
from core.agents.models.base import NNBase


class ActorMetaCritic(NNBase):
    """
        The actor-crictic network consists of meta-learner critic adapting 
            to different inputs sequence
    """
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super().__init__(recurrent, num_inputs, hidden_size)

        self.actor = nn.Sequential(
                linear_block(num_inputs, hidden_size),
                linear_block(hidden_size, hidden_size),
                linear_block(hidden_size, hidden_size),
                linear_block(hidden_size, hidden_size)
                )
        self.critic = MetaNet(num_inputs, 1, hidden_size)
        
    def forward(self, inputs, rnn_hxs, marks, state_dict=None):
        x = inputs
        
        if self.is_recurrent:
                x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        values = self.critic(x, state_dict)
        hidden_actor = self.actor(x)
        
        return values, hidden_actor, rnn_hxs

    def compute_critic_grad(self, inputs, rnn_hxs, marks, state_dict, gt, criterion):
        """
            Forward with fast weight and return grad
        """
        x = inputs 

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        value, grad = self.critic(x, state_dict, gt=gt, criterion=criterion)

        return grad 

    def clone_state_dict(self):
        """ Return the current state dict of critic network"""
        return self.critic.state_dict()

    
class MetaNet(nn.Module):
    """
        Meta learner that supports `forward` call with adapt parameters 
    """
    def __init__(self, in_channels, num_classes=1, hidden_size=64):
        super().__init__()
        self.in_channels = in_channels 
        self.out_channels = num_classes
        self.hidden_size = hidden_size

        self.layers = nn.Sequential(
                linear_block(self.in_channels, self.hidden_size),
                linear_block(self.in_channels, self.hidden_size),
                linear_block(self.in_channels, self.hidden_size),
                linear_block(self.in_channels, self.out_channels) 
               )
    
    def forward(self, inputs, state_dict=None, gt=None, criterion=None):
        if state_dict is None:
            out = self.layers(inputs)
        else:
            model = copy.deepcopy(self.layers)
            model.load_state_dict(state_dict)
            out = model(inputs)

        # if label and criterion are given, compute grad of fast weight
        if gt is None or criterion is None:
            return out 
        else:
            loss = criterion(out, gt)
            grad = torch.autograd.grad(loss, model.parameters(), create_graph=True)
            return out, grad


def linear_block(num_inputs, num_outputs):
    linear = nn.Sequential(
        nn.Linear(num_input, num_outputs),
        nn.BatchNorm1d(num_outputs),
        nn.Relu()
        )

    return linear
