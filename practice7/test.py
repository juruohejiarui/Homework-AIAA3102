from train import generate_data, init_random, get_device, Model, SelfDataset, MODEL_PATH
from torch.utils.data import DataLoader
from torch.nn import MSELoss as LossFn
from tqdm import tqdm, trange
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn

if __name__ == "__main__" :
    init_random(84)
    n = 200
    dataset = SelfDataset(*generate_data(n))
    
    model = Model()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(get_device())
    
    te_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    loss_fn = LossFn()
    
    model.eval()
    
    loss_sum = 0
    
    y_true = []
    y_pred = []
    
    with torch.no_grad() :
        for x, y in tqdm(te_loader) :
            x : torch.Tensor = x.to(get_device())
            y : torch.Tensor = y.to(get_device())
            pred : torch.Tensor = model(x)
            pred = pred.reshape(-1)
            loss : torch.Tensor = loss_fn(pred, y)
            
            loss_sum += loss.sum().item()
            
            y_true.extend(y.to("cpu").numpy().tolist())
            y_pred.extend(pred.to("cpu").numpy().tolist())
    
    print(f"Test MSE Loss: {loss_sum / n}")
    print(f"Mean sequare error: {mean_squared_error(y_true, y_pred)}")