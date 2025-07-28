import torch
from CLAP import get_audio_features, int16_to_float32, float32_to_int16

def extract_attention(clap_module, data, use_tensor=False):
    clap_module.model.eval()
    audio_input = []
    for audio_waveform in data:          
        # quantize
        if not use_tensor:
                #audio_waveform = int16_to_float32(float32_to_int16(audio_waveform))
                #audio_waveform = torch.from_numpy(audio_waveform).float()
                temp_dict = {}
                temp_dict = get_audio_features(
                    temp_dict, audio_waveform, 480000, 
                    data_truncating='fusion' if clap_module.enable_fusion else 'rand_trunc', 
                    data_filling='repeatpad',
                    audio_cfg=clap_module.model_cfg['audio_cfg'],
                    require_grad=audio_waveform.requires_grad
                )
                audio_input.append(temp_dict)

    out_dict = clap_module.model.get_audio_output_dict(audio_input)
    return out_dict
