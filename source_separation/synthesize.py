import numpy as np
import fire
import torch
import librosa
import source_separation
from pytorch_sound import settings
from pytorch_sound.utils.commons import get_loadable_checkpoint
from pytorch_sound.models import build_model
from pytorch_sound.utils.calculate import volume_norm_log_torch


def run(audio_file: str, out_path: str, model_name: str, pretrained_path: str):
    wav, sr = librosa.load(audio_file, sr=settings.SAMPLE_RATE)
    if wav.dtype != np.float32:
        wav = wav.astype(np.float32)

    # load model
    print('Load model ...')
    model = build_model(model_name).cuda()
    chk = torch.load(pretrained_path)['model']
    model.load_state_dict(get_loadable_checkpoint(chk))

    # make tensor wav
    wav = torch.FloatTensor(wav).unsqueeze(0).cuda()

    # do volume norm
    wav = volume_norm_log_torch(wav)

    # inference
    print('Inference ...')
    with torch.no_grad():
        out_wav = model(wav)[0].cpu().numpy()

    # save wav
    librosa.output.write_wav(out_path, out_wav, settings.SAMPLE_RATE)

    print('Finish !')


if __name__ == '__main__':
    fire.Fire(run)
