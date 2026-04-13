import torch

def transformer_lr_schedule(
    d_model, 
    step_num, 
    warmup_steps=4000
):
    """
    Implements the learning rate schedule from the original Transformer paper (Vaswani et al., 2017).
    
    :param d_model: The dimensionality of the model's embeddings (also known as the hidden dimension).
    :type d_model: int
    :param step_num: The current training step.
    :type step_num: int
    :param warmup_steps: The number of transformer schedule warmup steps. Set to 0 for a continuously decaying learning rate.
    :type warmup_steps: int

    :return: The learning rate at the current step.
    :rtype: float
    """
    if warmup_steps <= 0:
        step_num += 4000
        warmup_steps = 4000
    step_num = step_num + 1e-6  # avoid division by 0

    if type(step_num) == torch.Tensor:
        arg = torch.min(step_num ** -0.5, step_num * (warmup_steps ** -1.5))
    else:
        arg = min(step_num ** -0.5, step_num * (warmup_steps ** -1.5))

    return (d_model ** -0.5) * arg


def generate_causal_mask(
    seq_len: int, 
    device=None
):
    """
    Creates a causal mask with True values above the diagonal.
    
    :param seq_len: The length of the sequence for which to create the mask.
    :type seq_len: int
    :param device: The device on which to create the mask (e.g., 'cpu' or 'cuda').
    :type device: str

    :return: A (seq_len, seq_len) boolean tensor where True values indicate masked positions.
    :rtype: torch.Tensor
    """
    return torch.triu(
        torch.ones(seq_len, seq_len, device=device), 
        diagonal=1
    ).bool()
