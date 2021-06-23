import torch
from torch.utils.data import Dataset


class KanHope(Dataset):
    r"""An abstract class representing a :class:`Dataset`.

        All datasets that represent a map from keys to data samples should subclass
        it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
        data sample for a given key. Subclasses could also optionally overwrite
        :meth:`__len__`, which is expected to return the size of the dataset by many
        :class:`~torch.utils.data.Sampler` implementations and the default options
        of :class:`~torch.utils.data.DataLoader`.

        .. note::
          :class:`~torch.utils.data.DataLoader` by default constructs a index
          sampler that yields integral indices.  To make it work with a map-style
          dataset with non-integral indices/keys, a custom sampler must be provided.
        """

    def __init__(self, text: object, label: object, tokenizer: object, max_len: object) -> object:
        self.text = text
        self.label = label
        self.tokenizer = tokenizer
        self.max_len = max_len

    @property
    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        label = self.label[item]

        encoding1 = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'text': text,
            'input_ids1': encoding1['input_ids'].flatten(),
            'attention_mask1': encoding1['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float)
        }
