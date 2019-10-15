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
        'kernel_size': 3,
        'is_mask': True,
        'norm': 'bn',
        'act': 'comp'
    }


@register_model_architecture('refine_spectrogram_unet', 'refine_unet_base')
def refine_unet_base():
    return {
        'spec_dim': 256 + 1,
        'hidden_dim': 768,
        'filter_len': 512,
        'hop_len': 64,
        'block_layers': 4,
        'layers': 4,
        'kernel_size': 3,
        'refine_layers': 1,
        'is_mask': True,
        'norm': 'ins',
        'act': 'comp'
    }


@register_model_architecture('refine_spectrogram_unet', 'refine_unet_larger')
def refine_unet_larger():
    return {
        'spec_dim': 512 + 1,
        'hidden_dim': 768,
        'filter_len': 512 * 2,
        'hop_len': 64 * 2,
        'block_layers': 4,
        'layers': 4,
        'kernel_size': 3,
        'refine_layers': 1,
        'is_mask': True,
        'norm': 'ins',
        'act': 'comp'
    }


@register_model_architecture('refine_spectrogram_unet', 'refine_unet_larger_add')
def refine_unet_larger_add():
    d = refine_unet_larger()
    d.update(
        add_spec_results=True
    )
    return d
