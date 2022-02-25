from torch.utils.data import Dataset

class StudentData(Dataset):
    def __init__(self, dataset, targets):
        
        self.dataset = dataset
        self.targets = targets
        self.indices = dataset.indices
            
    def __getitem__(self, index):
        if isinstance(index, list):
            data, _ = self.dataset[[i for i in index]]
            target = self.targets[[self.indices[i] for i in index]]
        else:
            data, _ = self.dataset[index]
            target = self.targets[self.indices[index]]
            
        return data, target
        
    
    def __len__(self):
        """Total number of samples"""
        return len(self.indices)