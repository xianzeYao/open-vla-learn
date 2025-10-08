from torch.utils.data import IterableDataset
import numpy as np
class MyIterableDataset(IterableDataset):
    def __init__(self,start,end):
        super().__init__()
        self.start = start
        self.end = end
    def __iter__(self):
        i = np.arange(self.start,self.end)
        return iter(i)
if __name__ == "__main__":
    x = MyIterableDataset(1,9)
    for i in x:
        print(i)