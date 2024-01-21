
import torch

class sam_projection(torch.nn.Module):
    def __init__(self):
        super(sam_projection, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(256, 64, bias=True),
            torch.nn.LayerNorm(64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 64, bias=True),
            torch.nn.LayerNorm(64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, 32, bias=True)
        )

    def forward(self, x):
        return self.layers(x)


#sam_proj = sam_proj.cuda()
#sam_proj.train()
#param_group = {'params': sam_proj.parameters(), 'lr': opt.feature_lr, 'name': 'f'}
