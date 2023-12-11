from scipy.special import softmax
import time
import torch
import torch.utils.data
import torch.utils.data.dataset
import torch.autograd
import numpy as np

from gameEngine import TicTacToeMoves
Ms = TicTacToeMoves

class BaseNN(torch.nn.Module):
    def __init__(self, n_layers=3, hidden_size=32, n_iter=None, batch_size=None, lr=None):
        super().__init__()
        self.uid = time.strftime("%Y-%m-%d:%Hh%M")

        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.n_iter = 100 if n_iter is None else n_iter
        self.batch_size = 1024 if batch_size is None else batch_size
        self.lr = 0.001 if lr is None else lr

        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(9, hidden_size)
        ] + [
            torch.nn.Linear(hidden_size, hidden_size)
            for _ in range(n_layers - 1)
        ])
        self.evaluation_output = torch.nn.Linear(hidden_size, 1)
        self.policy_output = torch.nn.Linear(hidden_size, 9)

    def forward(self, x):
        x = x.float()
        x = x.reshape(*x.shape[:-2], -1)
        for layer in self.layers:
            x = torch.nn.functional.leaky_relu(layer(x))
        return self.evaluation_output(x).squeeze(dim=-1), self.policy_output(x)

    def predict(self, x):
        v, P = self(x)
        v = v.item()
        P = softmax(P.squeeze().detach().numpy())
        return v, P

    def fit(self, data, n_iter=None, batch_size=None, lr=None, progress=None):
        if n_iter is not None:
            self.n_iter = n_iter

        if batch_size is not None:
            self.batch_size = batch_size

        if lr is not None:
            self.lr = lr

        self.train()

        V, TP, TV = zip(*data)
        V = torch.stack(V)
        TP = torch.stack(TP)
        TV = torch.stack(TV)
        dataset = torch.utils.data.dataset.TensorDataset(V, TP, TV)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        def criterion(v, p, tv, tp):
            return (
                torch.nn.functional.mse_loss(v, tv),
                torch.nn.functional.cross_entropy(p, tp)
            )

        task = None
        if progress is not None:
            task = progress[0].add_task("Training", total=n_iter, arg1_n="loss", arg1="N/A")
        losses = []
        for _ in range(self.n_iter):
            cl = np.zeros(2)
            for s, tp, tv in loader:
                optimizer.zero_grad()
                v, p = self(s)
                loss = criterion(v, p, tv, tp)
                for i in range(2):
                    cl[i] += loss[i].item() * len(s)
                loss = loss[0] + loss[1]
                loss.backward()
                optimizer.step()
            cl /= len(data)
            losses.append(cl)
            if progress is not None:
                progress[0].update(task, advance=1, arg1=f'{cl.sum():.5f} = {cl[0]:.5f} + {cl[1]:.5f}')

        self.eval()

        return losses

    def id(self):
        return f'BaseNN-{self.n_layers}-{self.hidden_size}-{self.n_iter}-{self.batch_size}-{self.lr}_{self.uid}'





