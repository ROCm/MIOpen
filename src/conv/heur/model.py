import torch.nn as nn
import torch.nn.functional as F

class MIOpenNet(nn.Module):
  def __init__(self, num_features, num_solvers):
    super(MIOpenNet, self).__init__()
    self.fc1 = nn.Linear(num_features, 100);
    self.fc2 = nn.Linear(100, 200);
    self.fc3 = nn.Linear(200, 200);
    self.fc4 = nn.Linear(200, 200);
    self.fc5 = nn.Linear(200, 200);
    self.fc6 = nn.Linear(200, 100);
    self.fc7 = nn.Linear(100, num_solvers);
  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = F.relu(self.fc4(x))
    x = F.relu(self.fc5(x))
    x = F.relu(self.fc6(x))
    x = F.log_softmax(self.fc7(x), dim=1)
    return x
