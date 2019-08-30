import numpy as np
import os
from audioset_augmentor.augmentor import augment
from typing import List, Tuple, Any, Callable

from pytorch_sound.data.meta import MetaFrame
from pytorch_sound.data.meta.voice_bank import VoiceBankMeta
from pytorch_sound.data.dataset import SpeechDataset, SpeechDataLoader
from pytorch_sound.utils.sound import preemphasis


class AugmentSpeechDataset(SpeechDataset):

    def __init__(self, meta_frame: MetaFrame, is_audioset: bool = True, fix_len: int = 0, fix_shuffle: bool = False,
                 skip_audio: bool = False, audio_mask: bool = False, extra_features: List[Tuple[str, Callable]] = None):
        super().__init__(meta_frame, fix_len, fix_shuffle, skip_audio, audio_mask, extra_features)
        self.is_audioset = is_audioset

    def __getitem__(self, idx: int) -> List[Any]:
        res = super().__getitem__(idx)
        # augmentation with -1
        if np.random.randint(2):
            res[0] = res[0] * -1
            res[1] = res[1] * -1
        # augmentation with audioset data / twice on three times
        if self.is_audioset:
            if np.random.randint(3):
                rand_amp = np.random.rand() * 0.5 + 0.5
                res[0] = augment(res[1], amp=rand_amp)
        # do volume augmentation
        rand_vol = np.random.rand() + 0.5  # 0.5 ~ 1.5
        res[0] = np.clip(res[0] * rand_vol, -1, 1)
        res[1] = np.clip(res[1] * rand_vol, -1, 1)
        # pre emphasis
        res[0] = preemphasis(res[0])
        res[1] = preemphasis(res[1])
        return res


def get_datasets(meta_dir: str, batch_size: int, num_workers: int,
                 meta_cls: MetaFrame = VoiceBankMeta, is_audioset: bool = True,
                 fix_len: int = 0, skip_audio: bool = False,
                 audio_mask: bool = False) -> Tuple[SpeechDataLoader, SpeechDataLoader]:

    assert os.path.isdir(meta_dir), '{} is not valid directory path!'

    train_file, valid_file = VoiceBankMeta.frame_file_names[1:]

    # load meta file
    train_meta = meta_cls(os.path.join(meta_dir, train_file))
    valid_meta = meta_cls(os.path.join(meta_dir, valid_file))

    # create dataset
    train_dataset = AugmentSpeechDataset(train_meta, fix_len=fix_len, is_audioset=is_audioset,
                                         skip_audio=skip_audio, audio_mask=audio_mask)
    valid_dataset = AugmentSpeechDataset(valid_meta, fix_len=fix_len, is_audioset=is_audioset,
                                         skip_audio=skip_audio, audio_mask=audio_mask)

    # create data loader
    train_loader = SpeechDataLoader(train_dataset, batch_size=batch_size, is_bucket=False,
                                    num_workers=num_workers, skip_last_bucket=False)
    valid_loader = SpeechDataLoader(valid_dataset, batch_size=batch_size, is_bucket=False,
                                    num_workers=num_workers, skip_last_bucket=False)

    return train_loader, valid_loader
