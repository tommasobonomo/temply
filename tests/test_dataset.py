from torch.utils.data import DataLoader

from settings import Config
from src.dataset import PosDataset


def test_iteration_of_splits():
    config = Config()
    splits = list(config.split_paths.keys())

    for split in splits:
        dataset = PosDataset(config, split)
        dataloader = DataLoader(dataset, batch_size=32, num_workers=2)
        print(f"Looping over {split} dataloader...")
        for _ in dataloader:
            pass
