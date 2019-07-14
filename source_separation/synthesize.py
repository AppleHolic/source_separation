import numpy as np
import fire
import torch
import librosa
import source_separation
from scipy.io.wavfile import read as read_wav
from pytorch_sound import settings
from pytorch_sound.utils.commons import get_loadable_checkpoint
from pytorch_sound.models import build_model


def run(audio_file: str, out_path: str, model_name: str, pretrained_path: str):
    sr, wav = read_wav(audio_file)
    if wav.dtype != np.float32:
        wav = wav.astype(np.float32)

    # resample
    if sr != settings.SAMPLE_RATE:
        wav = librosa.core.resample(wav, sr, settings.SAMPLE_RATE)

    # load model
    model = build_model(model_name).cuda()
    chk = torch.load(pretrained_path)['model']
    model.load_state_dict(get_loadable_checkpoint(chk))

    # make tensor wav
    wav = torch.FloatTensor(wav).unsqueeze(0).cuda()

    # inference
    with torch.no_grad():
        out_wav = model(wav)[0].cpu().numpy()

    # save wav
    librosa.output.write_wav(out_path, out_wav, settings.SAMPLE_RATE)


if __name__ == '__main__':
    fire.Fire(run)
