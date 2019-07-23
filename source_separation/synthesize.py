import numpy as np
import fire
import torch
import librosa
import source_separation
from pytorch_sound import settings
from pytorch_sound.utils.commons import get_loadable_checkpoint
from pytorch_sound.models import build_model
from pytorch_sound.utils.calculate import volume_norm_log_torch
from pytorch_sound.utils.sound import lowpass
from pytorch_sound.models.sound import VolNormWindow


def run(audio_file: str, out_path: str, model_name: str, pretrained_path: str, is_norm: bool = True,
        lowpass_freq: int = 0):
    wav, sr = librosa.load(audio_file, sr=settings.SAMPLE_RATE)
    if wav.dtype != np.float32:
        wav = wav.astype(np.float32)

    # load model
    print('Load model ...')
    model = build_model(model_name).cuda()
    chk = torch.load(pretrained_path)['model']
    model.load_state_dict(get_loadable_checkpoint(chk))
    model.eval()

    # make tensor wav
    wav = torch.FloatTensor(wav).unsqueeze(0).cuda()

    # do volume norm
    if is_norm:
        normalizer = VolNormWindow(window_size=settings.SAMPLE_RATE // 4, target_db=settings.VN_DB)
        # wav = volume_norm_log_torch(wav)
        wav = normalizer.forward(wav)

    # inference
    print('Inference ...')
    with torch.no_grad():
        out_wav = model(wav)[0]
        if is_norm:
            out_wav = normalizer.reverse(out_wav)
        out_wav = out_wav.cpu().numpy()

    if lowpass_freq:
        out_wav = lowpass(out_wav, frequency=lowpass_freq)

    # save wav
    librosa.output.write_wav(out_path, out_wav, settings.SAMPLE_RATE)

    print('Finish !')


if __name__ == '__main__':
    fire.Fire(run)
