import os
import sys
dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dir_path)
from .hook import CLAP_Module
from .clap_module import htsat, model
from .training import get_audio_features, int16_to_float32, float32_to_int16