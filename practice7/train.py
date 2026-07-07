import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm, trange
from torch.optim import SGD as Optimizer
from torch.utils.data import DataLoader, Dataset

RANDOM_SEED = 42
MODEL_PATH = "model.pth"

def init_random(seed : int = RANDOM_SEED) : 
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def generate_data(n : int = 100) -> tuple[np.ndarray, np.ndarray] :
    def f(x) : return 3 * x[:, 0] + 4 * x[:, 1] - x[:, 2] ** 2
    X_1 = (np.random.random((n, 3)) - 0.5) * 10
    X_2 = (np.random.random((n, 3)) - 0.5) * 1
    X = np.concatenate([X_1, X_2])
    y = f(X)
    return X.astype(np.float32), y.astype(np.float32)

class Model(nn.Module) :
    def __init__(self) :
        super(Model, self).__init__()
        self.fc1 = nn.Linear(3, 32)
        self.loss1 = nn.Sigmoid()
        self.fc2 = nn.Linear(32, 32)
        self.loss2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, x : torch.Tensor) :
        x = self.loss1(self.fc1(x))
        x = self.loss2(self.fc2(x))
        return self.fc3(x)

class SelfDataset(Dataset) :
    def __init__(self, X : torch.Tensor, y : torch.Tensor) :
        self.X = X
        self.y = y
    def __len__(self) : return len(self.y)
    def __getitem__(self, index):
        return (self.X[index], self.y[index])
    
def get_device() -> str :
    if torch.cuda.is_available() : return "cuda"
    elif torch.mps.is_available() : return "mps"
    else: return "cpu"
    
def train(X_tr : torch.Tensor, y_tr : torch.Tensor, 
          epochs : int = 10,
          lr : float = 1e-2,
          save_path : str = MODEL_PATH) : 
    model = Model().to(device=get_device())
    model.train()
    dataset = SelfDataset(X_tr, y_tr)
    tr_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    optim = Optimizer(model.parameters())
    loss_fn = nn.MSELoss()
    for i in trange(epochs) :
        loss_sum = 0
        for x, y in tr_loader :
            x : torch.Tensor = x.to(get_device())
            y : torch.Tensor = y.to(get_device())
            pred : torch.Tensor = model(x)
            pred = pred.reshape(-1)
            loss : torch.Tensor = loss_fn(pred, y)
            loss.backward()
            
            with torch.no_grad() :
                loss_sum += loss.sum()
                
            optim.step()
            optim.zero_grad()
        
        tqdm.write(f"Epoch {i + 1:>3}: loss = {loss_sum.item() / len(dataset):.6f}")
    
    torch.save(model.state_dict(), save_path)

if __name__ == "__main__" :
    init_random()
    train(*generate_data(n=1000), epochs=100)