# import some packages you need here
import torch
import numpy as np

def generate(model, seed_characters, temperature, char_to_idx, idx_to_char, device, gen_length=100):
    """ Generate characters

    Args:
        model: trained model
        seed_characters: seed characters
        temperature: T
        char_to_idx: character to index mapping
        idx_to_char: index to character mapping
        device: device for computing, cpu or gpu
        gen_length: length of characters to generate

    Returns:
        samples: generated characters
    """
    
    model.eval()
    input_seq = torch.tensor([char_to_idx[ch] for ch in seed_characters], dtype=torch.long).unsqueeze(0).to(device)
    hidden = model.init_hidden(1)
    if isinstance(hidden, tuple):
        hidden = tuple(h.to(device) for h in hidden)
    else:
        hidden = hidden.to(device)

    samples = seed_characters
    
    with torch.no_grad():
        for _ in range(gen_length):
            output, hidden = model(input_seq, hidden)
            output = output / temperature
            probabilities = torch.softmax(output[-1], dim=0).cpu().numpy()
            next_idx = np.random.choice(len(probabilities), p=probabilities)
            next_char = idx_to_char[next_idx]
            
            samples += next_char
            
            input_seq = torch.tensor([[next_idx]], dtype=torch.long).to(device)

    return samples
