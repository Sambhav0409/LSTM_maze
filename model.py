import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuroMap(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, output_size=3):
        super(NeuroMap, self).__init__()
        self.hidden_size = hidden_size
        
        # LSTM for spatial memory
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # Action predictor
        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size))
        
    def forward(self, x, prev_state):
        # LSTM forward pass
        lstm_out, state = self.lstm(x, prev_state)
        
        # Add exploration noise during training
        if self.training:
            noise = torch.randn_like(lstm_out) * 0.1
            lstm_out = lstm_out + noise
        
        # Action probabilities
        action_logits = self.action_head(lstm_out.squeeze(1))
        action_probs = F.softmax(action_logits, dim=-1)
        
        return action_probs, state, None