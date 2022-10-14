from torch.utils.data import DataLoader

def CreateDataloadersDict(train, test, val=None, batch_size=100):
  dataloaders = {}
  dataloaders['train'] = DataLoader(train, batch_size=batch_size)
  dataloaders['test']  = DataLoader(test,  batch_size=batch_size)
  if val is not None: 
    dataloaders['val'] = DataLoader(val, batch_size=batch_size)
  return dataloaders

def GetDatasetSizes(dataloaders):
  dataset_sizes = {}
  for k in dataloaders.keys(): 
    dataset_sizes[k] = len(dataloaders[k].dataset)
  return dataset_sizes

