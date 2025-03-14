import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvNet(nn.Module):
    def __init__(self, n_embed_channel, size_kernal, dim_observe, size_world, n_rel, n_head=8):
        super().__init__()
        self.n_rel = n_rel
        self.dim_map = size_world[0] * size_world[1]
        self.dim_embed = n_embed_channel * ((dim_observe - size_kernal) + 1)**2 # dim_out of CNN = (dim_in - size_kernal + padding) / stride + 1
        self.encoder = nn.Conv2d(1, n_embed_channel, size_kernal)
        self.relations = nn.ModuleList([nn.MultiheadAttention(self.dim_embed, n_head) for _ in range(n_rel)])
        self.ffs = nn.ModuleList([nn.Linear(self.dim_embed, self.dim_embed) for _ in range(n_rel)])
        self.out = nn.Linear(self.dim_embed, self.dim_map)


    def forward(self, x, neighbors):
        '''
        x.shape = (1, dim_observe, dim_observe)
        neighbors.shape = (number of neighbors of current agent at this time step + 1(self), dim_embed)
        Using unbatch data because the number of neighbors changes with agent and time.
        '''
        x = self.encoder(x)
        x = x.view(1, self.dim_embed)
        for i in range(self.n_rel):
            x, _ = self.relations[i](x, neighbors, neighbors)
            x = F.relu(self.ffs[i](x))
            # Compared with transformer, no normalization here.
            out = torch.squeeze(self.out(x))
        return F.softmax(out, dim=0)
    
    def embed_observe(self, x):
        with torch.no_grad():
             x = self.encoder(x)
             x = x.view(1, self.dim_embed)
        return x


class GCNPos(nn.Module):
    def __init__(self, n_embed_channel, size_kernal, dim_observe, size_world, n_rel, n_head=8):
        super().__init__()
        self.n_rel = n_rel
        self.dim_map = size_world[0] * size_world[1]
        self.dim_embed = n_embed_channel * ((dim_observe - size_kernal) + 1)**2 # dim_out of CNN = (dim_in - size_kernal + padding) / stride + 1
        self.encoder = nn.Conv2d(1, n_embed_channel, size_kernal)
        self.linear = nn.Linear(self.dim_embed + 3, 512) # 3 extra extries represent the position and orientaion of the agent. 512 comes from "Attention is all you need".
        self.relations = nn.ModuleList([nn.MultiheadAttention(512, n_head) for _ in range(n_rel)])
        self.ffs = nn.ModuleList([nn.Linear(512, 512) for _ in range(n_rel)])
        self.out = nn.Linear(512, self.dim_map)


    def forward(self, x, neighbors, pos):
        '''
        x.shape = (1, dim_observe, dim_observe)
        pos.shape = (1, 3)
        neighbors.shape = (number of neighbors of current agent at this time step + 1(self), dim_embed)
        Using unbatch data because the number of neighbors changes with agent and time.
        '''
        if x.shape == (1, 6, 6):
            pass
        else:
            print(x.shape)
        x = self.encoder(x)
        x = x.view(1, self.dim_embed)
        x = torch.cat([x, pos], dim=-1)
        x = F.relu(self.linear(x))
        for i in range(self.n_rel):
            x, _ = self.relations[i](x, neighbors, neighbors)
            x = F.relu(self.ffs[i](x))
            # Compared with transformer, no normalization here.
            out = torch.squeeze(self.out(x))
        return F.softmax(out, dim=0)
    
    def embed_observe(self, x, pos):
        with torch.no_grad():
            if x.shape == (1, 6, 6):
                pass
            else:
                print(x.shape)
            x = self.encoder(x)
            x = x.view(1, self.dim_embed)
            x = torch.cat([x, pos], dim=-1)
            x = F.relu(self.linear(x))
        return x