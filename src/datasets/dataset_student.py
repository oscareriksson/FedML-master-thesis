from torch.utils.data import Dataset

class StudentData(Dataset):
    def __init__(self, dataset):
        
        self.dataset = dataset
            
    def __getitem__(self, index):
        if isinstance(index, list):
            data, _ = self.dataset[[i for i in index]]
        else:
            data, _ = self.dataset[index]
            
        return data, index
        
    
    def __len__(self):
        """Total number of samples"""
        return len(self.dataset)