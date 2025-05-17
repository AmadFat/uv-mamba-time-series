import torch

class Normalize:
    def __call__(self, latlon):
        return latlon / torch.tensor([90, 180], dtype=latlon.dtype)

class MPTSDataset(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None):
        from pathlib import Path
        import numpy as np
        import scipy.io

        root = Path(root) / ("train" if train else "test")
        self.mats = sorted(root.glob("*.mat"))
        self.tracks = []
        for mat in self.mats:
            assert mat.is_file()
            datas = scipy.io.loadmat(str(mat))["latlons"]
            for i, d in np.ndenumerate(datas):
                self.tracks.append(d[0])
        self.transform = transform
        
    def __len__(self):
        return len(self.tracks)
    
    def __getitem__(self, idx):
        x = torch.from_numpy(self.tracks[idx]).float()
        if self.transform is not None:
            x = self.transform(x)
        return x
    
    def split(self, ratios):
        assert sum(ratios) == 1
        return torch.utils.data.random_split(self, ratios)

def collate_fn(batch):
    x = torch.nested.as_nested_tensor(batch, layout=torch.jagged)
    coors = x.to_padded_tensor(padding=0)
    masks = x.bool().to_padded_tensor(padding=False)
    return coors, masks

def MPTSLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    drop_last=True,
    collate_fn=collate_fn,
):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )
    return loader

if __name__ == "__main__":
    dataset = MPTSDataset("./data", train=True, transform=Normalize())
    train_dataset, val_dataset = dataset.split([0.8, 0.2])
    print(len(train_dataset), len(val_dataset))
    loader = MPTSLoader(train_dataset, batch_size=2, shuffle=True)
    for i, (coors, masks) in enumerate(loader):
        print(coors.shape, masks.shape)
        if i == 2:
            break