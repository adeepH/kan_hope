import torch
from torch.utils.data import Dataset, DataLoader


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
    def __init__(self, text: object, translation: object, label: object, tokenizer1: object, tokenizer2: object, max_len: object) -> object:
        self.text = text
        self.translation = translation
        self.label = label
        self.tokenizer1 = tokenizer1
        self.tokenizer2 = tokenizer2
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        translation = str(self.translation[item])
        label = self.label[item]

        encoding1 = self.tokenizer1.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        encoding2 = self.tokenizer2.encode_plus(
            translation,
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
            'translation': translation,
            'input_ids1': encoding1['input_ids'].flatten(),
            'input_ids2': encoding2['input_ids'].flatten(),
            'attention_mask1': encoding1['attention_mask'].flatten(),
            'attention_mask2': encoding2['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float)
        }


