# import some packages you need here
import torch
from torch.utils.data import Dataset

class Shakespeare(Dataset):
    """ Shakespeare dataset

        To write custom datasets, refer to
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

    Args:
        input_file: txt file

    Note:
        1) Load input file and construct character dictionary {index:character}.
					 You need this dictionary to generate characters.
				2) Make list of character indices using the dictionary
				3) Split the data into chunks of sequence length 30. 
           You should create targets appropriately.
    """

    def __init__(self, input_file):

        # write your codes here
        # Load the input file
        with open(input_file, 'r') as f:
            self.text = f.read()
        
        # Construct character dictionary
        self.chars = sorted(list(set(self.text)))
        self.char_to_idx = {ch: idx for idx, ch in enumerate(self.chars)}
        self.idx_to_char = {idx: ch for idx, ch in enumerate(self.chars)}
        
        # Convert text to indices
        self.data = [self.char_to_idx[ch] for ch in self.text]
        
        # Define sequence length
        self.seq_length = 30

    def __len__(self):

        # write your codes here
        return len(self.data) - self.seq_length


    def __getitem__(self, idx):

        # write your codes here
        # Input sequence
        input_seq = self.data[idx:idx + self.seq_length]
        input = torch.tensor(input_seq, dtype=torch.long)
        # Target sequence (shifted by one)
        target_seq = self.data[idx + 1:idx + self.seq_length + 1]
        target = torch.tensor(target_seq, dtype=torch.long)



        return input, target

if __name__ == '__main__':

    # write test codes to verify your implementations
    dataset = Shakespeare('shakespeare_train.txt')
    
    # Print some details about the dataset
    print(f"Total sequences: {len(dataset)}")
    print(f"Example input sequence (as indices): {dataset[0][0]}")
    print(f"Example target sequence (as indices): {dataset[0][1]}")
    
    # Print the corresponding characters for the first example
    example_input_indices = dataset[0][0].numpy()
    example_target_indices = dataset[0][1].numpy()
    
    example_input_chars = ''.join([dataset.idx_to_char[idx] for idx in example_input_indices])
    example_target_chars = ''.join([dataset.idx_to_char[idx] for idx in example_target_indices])
    
    print(f"Example input sequence (as characters): {example_input_chars}")
    print(f"Example target sequence (as characters): {example_target_chars}")