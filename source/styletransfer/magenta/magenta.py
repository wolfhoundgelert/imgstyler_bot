# Source: https://arxiv.org/abs/1705.06830,
#   https://towardsdatascience.com/fast-neural-style-transfer-in-5-minutes-with-tensorflow-hub-magenta-110b60431dcc
#   https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2


import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

from source.styletransfer.styletransfer import \
    StyleTransfer, StyleTransferType, StyleTransferConfig, StyleTransferInference


class MagentaConfig(StyleTransferConfig):

    def __init__(self,
                 image_size: int = 512,
                 keep_content_aspect_ratio: bool = True,
                 ):
        super().__init__(image_size, keep_content_aspect_ratio)


class MagentaInference(StyleTransferInference):

    @staticmethod
    def _image_to_tensor(image):
        img = tf.convert_to_tensor(image)
        # Convert image to dtype, scaling (MinMax Normalization) its values if needed.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # Scale the image using the custom function we created

        # Adds a fourth dimension to the Tensor because the model requires a 4-dimensional Tensor
        img = img[tf.newaxis, :]
        return img

    @staticmethod
    def _tensor_to_image(tensor):
        tensor = tensor * 255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return Image.fromarray(tensor)

    _hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    def __init__(self, config: MagentaConfig, content_image: Image, style_image: Image):
        super().__init__(config, content_image, style_image)

    def _inference(self) -> Image:
        self._prepare_images()
        content = MagentaInference._image_to_tensor(self._content_image)
        style = MagentaInference._image_to_tensor(self._style_image)

        tf.keras.backend.clear_session()  # There were some issues with colors without this code

        result = MagentaInference._hub_module(tf.constant(content), tf.constant(style))[0]
        return MagentaInference._tensor_to_image(result)


class Magenta(StyleTransfer):

    def __init__(self, config: MagentaConfig = None):
        super().__init__(config)

    def get_default_config(self):
        return MagentaConfig()

    def get_inference(self, content_image: Image, style_image: Image, config: MagentaConfig = None):
        return MagentaInference(config or self._config, content_image, style_image)

    def type(self):
        return StyleTransferType.Magenta
