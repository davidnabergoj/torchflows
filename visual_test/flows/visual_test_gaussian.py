import torch
import torch.optim as optim
from normalizing_flows import Flow, RealNVP, NICE, MAF, IAF
import matplotlib.pyplot as plt


def visual_test_basic(bijection_class):
    n_dim = 2
    n_samples = 1000
    scale = torch.tensor([1.0, 10.0])
    n_epochs = 100
    device = torch.device('cpu')

    torch.manual_seed(0)
    bijection = bijection_class(n_dim=n_dim, n_layers=3)
    flow = Flow(bijection).to(device)
    x = torch.randn(n_samples, n_dim) * scale.view(1, -1)
    x = x.to(device)

    flow.sample(1000)

    log_prob = flow.log_prob(x)
    assert log_prob.shape == (n_samples,)

    optimizer = optim.AdamW(flow.parameters())
    for i in range(n_epochs):
        optimizer.zero_grad()
        loss = -flow.log_prob(x).mean()
        loss.backward()
        optimizer.step()
        # print(f'[{i}] Loss: {float(loss):.4f}')

    with torch.no_grad():
        plt.figure()
        y = flow.sample(1000)
        y = y.cpu()
        x = x.cpu()
        plt.scatter(y[:, 0], y[:, 1], label='Flow')
        plt.scatter(x[:, 0], x[:, 1], label='Train')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    for bijection_class in [NICE, RealNVP, MAF, IAF]:
        print(bijection_class)
        visual_test_basic(bijection_class)
