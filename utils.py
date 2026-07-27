def torch_to_numpy(a):
    """
    Converts a torch tensor to a numpy array (detached, on CPU).
    """
    return a.detach().cpu().numpy()
