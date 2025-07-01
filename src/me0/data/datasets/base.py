import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor
import tensordict as td
from tensordict import TensorDict
from torch.utils.data import Dataset


#def get_total_entries(paths: dict[str, str]) -> int:
#    return sum(each for _, _, each in uproot.num_entries(paths))

def get_total_entries(tree_iter, nfiles) -> int:
    entries = [len(tree_iter[each]) for each in tree_iter]
    return sum(entries[:nfiles])

def pad_muon(chunk: npt.NDArray[np.object_], dtype=torch.float32) -> Tensor:
    output = [torch.from_numpy(each) for each in chunk]
    return torch.nested.nested_tensor(output, dtype=dtype).to_padded_tensor(padding=-1)

class TensorDictListDataset(Dataset):

    def __init__(self, data: list[TensorDict]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> TensorDict:
        return self.data[index]

    def __add__(self, # type: ignore[override]
                other: 'TensorDictListDataset'
    ) -> 'TensorDictListDataset':
        data = self.data + other.data
        return self.__class__(data)

    def collate(self, batch: list[TensorDict]) -> TensorDict:
        batch = td.pad_sequence(
            list_of_tensordicts=batch,
            pad_dim=0,
            padding_value=-1,
            return_mask=False,
        )
        return batch
