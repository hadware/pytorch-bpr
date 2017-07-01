

class DatasetSplit:
    pass


class Dataset:

    def __init__(self, trainset_size : int, validationset_size : int, testst_size : int):
        pass

    def get_dataset_split(self):
        pass


class CFDataset(Dataset):
    """Dataset corresponding to collaborative filtering data.
    Only positive ratings are avalaible ("likes" on a post for instance)"""
    pass

class RankedDataset(Dataset):
    """Dataset corresponding to """