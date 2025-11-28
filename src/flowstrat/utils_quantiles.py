import torch


def simulate_from_flow(model, size, dim=1):
    N_sample = torch.normal(0, 1, size=(size, dim))
    sample = model(N_sample, None, reverse=True).cpu().detach().numpy()
    return sample


def transform_by_flow(model, value):
    value = torch.tensor(value)
    value = value.type(torch.FloatTensor)
    return model(value, None, reverse=True).cpu().detach().numpy()
