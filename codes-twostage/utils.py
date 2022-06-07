import torch
import numpy as np

def save_model(model_path, model, optimizer):
    try:
        state = {'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict()}
    except:
        state = {'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()}
    torch.save(state, model_path)
    print('model saved to {}'.format(model_path))

def load_model(model_path, model, device=torch.device('cuda'), optimizer=None):
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    print('model loaded from {}'.format(model_path))
    return model, optimizer


