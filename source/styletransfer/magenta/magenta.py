# Source: https://arxiv.org/abs/1705.06830,
#   https://towardsdatascience.com/fast-neural-style-transfer-in-5-minutes-with-tensorflow-hub-magenta-110b60431dcc
#   https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2


from numpy import array, uint8, ndim
from tensorflow.python.ops.image_ops_impl import convert_image_dtype
from tensorflow import convert_to_tensor, newaxis, float32, constant
from keras.backend import clear_session
from tensorflow_hub import load

from PIL import Image

from styletransfer.styletransfer import \
    StyleTransfer, StyleTransferType, StyleTransferConfig, StyleTransferInference


class MagentaConfig(StyleTransferConfig):

    def __init__(self,
                 image_size: int = 256,
                 keep_content_aspect_ratio: bool = True,
                 ):
        super().__init__(image_size, keep_content_aspect_ratio)


class MagentaInference(StyleTransferInference):

    @staticmethod
    def _image_to_tensor(image):
        img = convert_to_tensor(image)
        # Convert image to dtype, scaling (MinMax Normalization) its values if needed.

        img = convert_image_dtype(img, float32)
        # Scale the image using the custom function we created

        # Adds a fourth dimension to the Tensor because the model requires a 4-dimensional Tensor
        img = img[newaxis, :]
        return img

    @staticmethod
    def _tensor_to_image(tensor):
        tensor = tensor * 255
        tensor = array(tensor, dtype=uint8)
        if ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return Image.fromarray(tensor)

    _hub_module = load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

    def __init__(self, config: MagentaConfig, content_image: Image, style_image: Image):
        super().__init__(config, content_image, style_image)

    def _inference(self) -> Image:
        self._prepare_images()
        content = MagentaInference._image_to_tensor(self._content_image)
        style = MagentaInference._image_to_tensor(self._style_image)

        clear_session()  # There were some issues with colors without this code

        result = MagentaInference._hub_module(constant(content), constant(style))[0]
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
