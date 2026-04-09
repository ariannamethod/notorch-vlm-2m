"""
Train a vlm — notorch edition
Chuck replaces Adam. Adam stays as fallback. Chuck sees. Adam is blind.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ariannamethod.notorch_py import notorch as torch
from ariannamethod.chuck import ChuckOptimizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_model(model, optimizer, train_dataloader, n_epochs):
    model = model.to(device)
    model.train()
    
    for epoch in range(n_epochs):
        losses = []
        for bi, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            
            batch['image'] = batch['image'].to(device)
            
            logits, loss = model.forward(batch)
            loss.backward()
            optimizer.step(loss=loss.item())  # Chuck needs the loss
            
            losses.append(loss.data.item())
            if bi % 100 == 0:
                avg_loss = sum(losses) / len(losses)
                print(f'epoch {epoch} step {bi} | avg loss: {avg_loss:.4f}')
        
        avg_loss = sum(losses) / len(losses)
        print(f'epoch {epoch} done | avg loss: {avg_loss:.4f}')

if __name__ == '__main__':
    """
    Train a model — Chuck optimizer (Adam fallback)
    """
    
    # Model
    from vlm import build_vlm

    model = build_vlm().to(device)
    print(model)

    # Optimizer — Chuck (Adam as fallback)
    try:
        optimizer = ChuckOptimizer(model.parameters(), lr=3e-4)
        print("Using Chuck Optimizer — Adam is blind. Chuck sees.")
    except Exception as e:
        print(f"Chuck unavailable ({e}), falling back to AdamW")
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # Data
    from dataset import get_coco_dataset
    from torch.utils.data import DataLoader

    train_dataset = get_coco_dataset(mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=1)
    
    val_dataset = get_coco_dataset(mode='val')
    val_dataloader = DataLoader(val_dataset, batch_size=1)

    print(train_dataset, val_dataset)
    n_epochs = 2
    
    train_model(model, optimizer, train_dataloader, n_epochs)
