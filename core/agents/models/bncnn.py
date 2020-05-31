from torch import nn
from core.agents.models.base import NNBase, Flatten
from core.utils import init

class BNCNN(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super().__init__(recurrent, hidden_size, hidden_size)

        # shared weight that encode state to vector
        self.main = nn.Sequential(
            self.init_weight(nn.Conv2d(num_inputs, 32, 4, stride=2, bias=False)), 
            self.init_weight(nn.BatchNorm2d(32)), # 32x41x41
            nn.ReLU(), 
            self.init_weight(nn.Conv2d(32, 64, 5, stride=2, bias=False)), 
            self.init_weight(nn.BatchNorm2d(64)), # 64x19x19
            nn.ReLU(), 
            self.init_weight(nn.Conv2d(64, 128, 3, stride=2, bias=False)), 
            self.init_weight(nn.BatchNorm2d(128)), # 128x9x9
            nn.ReLU(), 
            self.init_weight(nn.Conv2d(128, 256, 5, stride=2, bias=False)), 
            self.init_weight(nn.BatchNorm2d(256)), 
            nn.ReLU(), 
            Flatten(), # 256x3x3
            self.init_weight(nn.Linear(256*3*3, hidden_size)), 
            nn.ReLU())

        self.critic_linear = nn.Sequential(
            self.init_weight(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            self.init_weight(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            self.init_weight(nn.Linear(hidden_size, 1))
        )

        # encoder for learning contrastive predictive objective
        self.contrastive_encoder = nn.Sequential(
            self.init_weight(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            self.init_weight(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            self.init_weight(nn.Linear(hidden_size, hidden_size))
        )
        self.train()
    
    def init_weight(self, layer):
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            return init_(layer)
        elif isinstance(layer, nn.BatchNorm2d):
            layer.weight.data.fill_(1)
            if hasattr(layer, 'bias'):
                layer.bias.data.zero_()
            return layer 

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), self.contrastive_encoder(x), rnn_hxs
