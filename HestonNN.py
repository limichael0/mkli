import torch.nn as nn

class Net(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, ac_fun='ReLU'):
        super(Net,self).__init__()
        self.h1 = nn.Linear(n_input, n_hidden)
        self.h2 = nn.Linear(n_hidden, n_hidden)
        self.h3 = nn.Linear(n_hidden, n_hidden)
        self.h4 = nn.Linear(n_hidden, n_hidden)
        self.fl = nn.Linear(n_hidden, n_output)
        
        if ac_fun == 'PReLU':
            self.ac1 = nn.PReLU(n_hidden)
            self.ac2 = nn.PReLU(n_hidden)
            self.ac3 = nn.PReLU(n_hidden)
            self.ac4 = nn.PReLU(n_hidden)
            self.ac5 = nn.PReLU(n_output)
        elif ac_fun == 'ELU':
            self.ac1 = nn.ELU()
            self.ac2 = nn.ELU()
            self.ac3 = nn.ELU()
            self.ac4 = nn.ELU()
            self.ac5 = nn.ELU()
        elif ac_fun == 'Softplus':
            self.ac1 = nn.Softplus()
            self.ac2 = nn.Softplus()
            self.ac3 = nn.Softplus()
            self.ac4 = nn.Softplus()
            self.ac5 = nn.Softplus()
        else: #ReLU
            self.ac1 = nn.ReLU()
            self.ac2 = nn.ReLU()
            self.ac3 = nn.ReLU()
            self.ac4 = nn.ReLU()
            self.ac5 = nn.ReLU()
        
    def forward(self,x):
        x = self.ac1(self.h1(x))
        x = self.ac2(self.h2(x))
        x = self.ac3(self.h3(x))
        x = self.ac4(self.h4(x))
        x = self.ac5(self.fl(x))
        return x