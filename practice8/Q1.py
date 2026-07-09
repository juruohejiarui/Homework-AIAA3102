import torch
import torch.nn as nn
import random
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import StepLR as Scheduler
from torch.nn import CrossEntropyLoss as Criterion
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

MODEL_PATH = "./mymodel.pth"
DATA_PATH = "./data"

def get_device() -> str :
    try :
        if torch.cuda.is_available() : return "cuda"
        elif torch.mps.is_available() : return "mps"
        else: return "cpu"
    except Exception as e :
        return "cpu"
    
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


class Component(nn.Module) :
    def __init__(self, in_channels : int, out_channels : int, kernel_sz : int, stride : int = 1, padding : int = 0) :
        super().__init__()
        self.blk = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_sz, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x : torch.Tensor) -> torch.Tensor :
        return self.blk(x)
        
class Model(nn.Module) :
    # from content of lecture slides and reference of other implementation of resnet 
    class ResnetBlk(nn.Module) :
        def __init__(self, in_channels : int, out_channels : int, stride : int) :
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
            
            if stride != 1 or in_channels != out_channels :
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            else :
                self.shortcut = nn.Identity()
                
            self.relu = nn.ReLU()
        
        def forward(self, x : torch.Tensor) -> torch.Tensor :
            res : torch.Tensor = self.shortcut(x)
            
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.bn2(self.conv2(x))
            
            return self.relu(x + res)
    
    class InceptionModule(nn.Module) :
        def __init__(self, in_channels : int, out_channels : int) :
            super().__init__()
            
            b1 = out_channels // 4
            b2 = out_channels - (b1 * 3)

            self.comp1 = Component(in_channels, b1, kernel_sz=1)
            self.comp2 = nn.Sequential(
                Component(in_channels, b1, 1),
                Component(b1, b1, 3, padding=1)
            )
            self.comp3 = nn.Sequential(
                Component(in_channels, b1, 1),
                Component(b1, b1, 5, padding=2)
            )
            self.comp4 = nn.Sequential(
                nn.MaxPool2d(3, 1, padding=1),
                Component(in_channels, b2, 1)
            )
        def forward(self, x : torch.Tensor) :
            o1 = self.comp1(x)
            o2 = self.comp2(x)
            o3 = self.comp3(x)
            o4 = self.comp4(x)

            return torch.cat([o1, o2, o3, o4], 1)
        
    def __init__(self, num_lbls : int = 10, drop_p : float = 0.3) :
        super(Model, self).__init__()
        
        self.conv = nn.Sequential(
            Component(3, 64, 3, stride=1, padding=1),
            self.ResnetBlk(64, 64, stride=1),
            self.ResnetBlk(64, 64, stride=1),
            self.InceptionModule(64, 128),
            
            self.ResnetBlk(128, 128, stride=2),
            self.ResnetBlk(128, 128, stride=1),
            self.InceptionModule(128, 256),

            self.ResnetBlk(256, 256, stride=2),
            self.ResnetBlk(256, 256, stride=1)
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(drop_p)
        self.classifier = nn.Linear(256, num_lbls)

    def forward(self, x : torch.Tensor) :
        x = self.conv(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        x = self.classifier(x)
        return x

def train_one_epoch(model : nn.Module, loader : DataLoader, criterion : Criterion, optimizer : Optimizer, dev : str) -> tuple[float, float] :
    model.train()
    loss_sum = 0.0
    num_acc = 0
    num_samples = 0

    for imgs, lbls in loader :
        imgs : torch.Tensor = imgs.to(dev, non_blocking=True)
        lbls : torch.Tensor = lbls.to(dev, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        prob : torch.Tensor = model(imgs)
        loss : torch.Tensor = criterion(prob, lbls)
        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        loss_sum += loss.item() * bs
        num_samples += bs

        num_acc += (prob.argmax(dim=1) == lbls).sum().item()
    
    return (loss_sum / num_samples, num_acc / num_samples)

def eval(model : nn.Module, loader : DataLoader, dev : str) -> float :
    model.eval()

    num_acc = 0
    num_samples = 0

    with torch.no_grad() :
        for imgs, lbls in loader :
            imgs : torch.Tensor = imgs.to(dev, non_blocking=True)
            lbls : torch.Tensor = lbls.to(dev, non_blocking=True)

            prob : torch.Tensor = model(imgs)
            
            bs = imgs.size(0)
            
            num_acc += (prob.argmax(dim=1) == lbls).sum().item()
            num_samples += bs
    
    return num_acc / num_samples

def get_loaders(data_path : str = DATA_PATH, batch_sz : int = 32, num_workers : int = 4) -> tuple[DataLoader, DataLoader] :
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2023, 0.1994, 0.2010)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])

    train_set = CIFAR10(root=data_path, train=True, download=True, transform=train_transform)
    test_set = CIFAR10(root=data_path, train=False, download=True, transform=test_transform)

    return ( \
        DataLoader(train_set, batch_sz, shuffle=True, num_workers=num_workers, pin_memory=True),
        DataLoader(test_set, batch_sz, shuffle=False, num_workers=num_workers, pin_memory=True))

def save_model(model : nn.Module, path : str = MODEL_PATH) :
    torch.save(model.state_dict(), path)

def load_model(model_class, dev : str, path : str = MODEL_PATH) :
    model : nn.Module = model_class().to(dev)
    model.load_state_dict(torch.load(path, map_location=dev))
    return model

def load_pretrained_mobilenetv2_x0_5(device):
    """
    Load the pretrained CIFAR-10 MobileNetV2 x0.5 from the repository.
    The repo documents torch.hub usage and lists mobilenetv2_x0_5 in the CIFAR-10 zoo.
    """
    candidate_names = [
        "cifar10_mobilenetv2_x0_5",
        "mobilenetv2_x0_5",
    ]

    last_err = None
    for name in candidate_names:
        try:
            model = torch.hub.load("chenyaofo/pytorch-cifar-models", name, pretrained=True)
            model = model.to(device)
            model.eval()
            return model, name
        except Exception as e:
            last_err = e

    raise RuntimeError(
        f"Failed to load pretrained MobileNetV2 x0.5 from torch.hub. "
        f"Tried {candidate_names}. Last error: {last_err}"
    )

if __name__ == "__main__" :
    seed_everything()

    dev = get_device()
    print(f"Use device '{dev}'")

    tr_loader, te_loader = get_loaders(batch_sz=128, num_workers=8)

    model = Model().to(dev)
    criterion = Criterion()
    optimizer = Optimizer(model.parameters(), lr=1e-3)
    scheduler = Scheduler(optimizer, step_size=5, gamma=0.5)

    print(f"Training...")
    for epoch in trange(30) :
        tr_loss, tr_acc = train_one_epoch(model, tr_loader, criterion, optimizer, dev)
        te_acc = eval(model, te_loader, dev)
        tqdm.write(f"Epoch {epoch + 1}: loss: {tr_loss:.4f} train acc: {tr_acc:.4f} test acc: {te_acc:.4f} lr: {optimizer.param_groups[0]["lr"]}")

        scheduler.step()

    save_model(model)
    print(f"Save model to {MODEL_PATH}")

    model = load_model(Model, dev)
    te_acc = eval(model, te_loader, dev)
    print(f"Model '{MODEL_PATH}' test acc: {te_acc:.4f}")

    pretrained_model, name = load_pretrained_mobilenetv2_x0_5(dev)
    te_acc = eval(pretrained_model, te_loader, dev)
    print(f"Model '{name}' test acc: {te_acc:.4f}")
