from pytorch_sound.models import register_model_architecture


@register_model_architecture('spectrogram_unet', 'spectrogram_unet_comp')
def spec_unet_comp():
    return {
        'spec_dim': 513,
        'hidden_dim': 384,
        'filter_len': 1024,
        'hop_len': 256,
        'block_layers': 4,
        'layers': 5,
        'kernel_size': 5,
        'is_mask': True,
        'norm': 'bn',
        'act': 'comp'
    }


@register_model_architecture('spectrogram_unet', 'spectrogram_unet_comp_thin')
def spec_unet_comp():
    return {
        'spec_dim': 513,
        'hidden_dim': 256,
        'filter_len': 1024,
        'hop_len': 256,
        'block_layers': 3,
        'layers': 4,
        'kernel_size': 5,
        'is_mask': True,
        'norm': 'bn',
        'act': 'comp'
    }


@register_model_architecture('refine_spectrogram_unet', 'refine_spectrogram_unet_base')
def spec_unet_comp():
    return {
        'spec_dim': 513,
        'hidden_dim': 256,
        'filter_len': 1024,
        'hop_len': 256,
        'block_layers': 3,
        'layers': 4,
        'kernel_size': 5,
        'is_mask': True,
        'norm': 'bn',
        'act': 'comp'
    }
