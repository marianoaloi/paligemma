import os
# from google.colab import userdata

# os.environ["KAGGLE_USERNAME"] = userdata.get('KAGGLE_USERNAME')
# os.environ["KAGGLE_KEY"] = userdata.get('KAGGLE_KEY')

import sys


if "COLAB_TPU_ADDR" in os.environ:
  raise "It seems you are using Colab with remote TPUs which is not supported."

if not os.path.exists("big_vision_repo"):
    pass
#   !git clone --quiet --branch=main --depth=1 \
#      https://github.com/google-research/big_vision big_vision_repo

if "big_vision_repo" not in sys.path:
  sys.path.append("big_vision_repo")

# Install missing dependencies. Assume jax~=0.4.25 with GPU available.
# !pip3 install -q "overrides" "ml_collections" "einops~=0.7" "sentencepiece"


import base64
import functools
import html
import io
import os
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import ml_collections

import tensorflow as tf
import sentencepiece

from IPython.core.display import display, HTML
from PIL import Image

# Import model definition from big_vision
from big_vision.models.proj.paligemma import paligemma
from big_vision.trainers.proj.paligemma import predict_fns

# Import big vision utilities
import big_vision.datasets.jsonl
import big_vision.utils
import big_vision.sharding

# Don't let TF use the GPU or TPUs
tf.config.set_visible_devices([], "GPU")
tf.config.set_visible_devices([], "TPU")

backend = jax.lib.xla_bridge.get_backend()
print(f"JAX version:  {jax.__version__}")
print(f"JAX platform: {backend.platform}")
print(f"JAX devices:  {jax.device_count()}")