import torch

from normalizing_flows import Flow, MAF
from potentials import GaussianMixture2D
import torch.optim as optim
import matplotlib.pyplot as plt

if __name__ == '__main__':
    torch.manual_seed(0)
    flow = Flow(MAF(n_dim=2, n_layers=3)).cuda()
    pot = GaussianMixture2D()
    x = pot.sample((10000,))
    optimizer = optim.AdamW(flow.parameters())
    for e in range(100000):
        optimizer.zero_grad()
        loss = -flow.log_prob(x.cuda()).mean()
        loss.backward()
        optimizer.step()
        loss = float(loss)
        print(f'[{e}] {loss = }')

    with torch.no_grad():
        fig, ax = plt.subplots()
        xt = pot.sample((10000,))
        ax.scatter(xt[:, 0], xt[:, 1], label='Train')
        xs = flow.cpu().sample(10000)
        ax.scatter(xs[:, 0], xs[:, 1], s=5, label='Flow')
        ax.legend()
        plt.show()
