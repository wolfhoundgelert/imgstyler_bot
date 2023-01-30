from copy import deepcopy
from PIL import Image


class StyleTransferType:

    Gatys = 'Gatys'  # https://arxiv.org/abs/1508.06576 - nice but slow
    MSGNet = 'MSGNet'  # https://arxiv.org/abs/1703.06953 - the fastest one, but sometimes with visible patterns
    Magenta = 'Magenta'  # https://arxiv.org/abs/1705.06830 - fast enough
    MSGNetCustomTrain = 'MSGNetCustomTrain'  # TODO link on my jupyter notebook on github


class StyleTransferConfig:

    def __init__(self,
                 image_size: int = 512,
                 keep_content_aspect_ratio: bool = True
                 ):

        self.image_size = image_size
        self.keep_content_aspect_ratio = keep_content_aspect_ratio


class StyleTransferInference:

    @staticmethod
    def _convert_to_rgb(image: Image):
        return image.convert('RGB') if image.format != 'RGB' else image

    def __init__(self, config: StyleTransferConfig, content_image: Image, style_image: Image):
        self._config = config
        self._content_image = content_image
        self._style_image = style_image

    def _resize_image(self, image: Image, ref: Image = None):
        if ref:
            w, h = ref.size
            return image.resize((w, h), Image.Resampling.LANCZOS)

        size = self._config.image_size
        if self._config.keep_content_aspect_ratio:
            w, h = image.size
            ar = w / h
            w = size if ar >= 1 else int(size * ar)
            h = size if ar <= 1 else int(size / ar)
            image = image.resize((w, h), Image.Resampling.LANCZOS)
        else:
            image = image.resize((size, size), Image.Resampling.LANCZOS)
        return image

    def _prepare_images(self):
        content_image = self._convert_to_rgb(self._content_image)
        style_image = self._convert_to_rgb(self._style_image)

        content_image = self._resize_image(content_image)
        style_image = self._resize_image(style_image, ref=content_image)

        self._content_image = content_image
        self._style_image = style_image

    def _inference(self) -> Image:
        assert False, print("Should be overriden")

    # Defining __call__ method
    def __call__(self):
        return self._inference()


class StyleTransfer:

    def __init__(self, config: StyleTransferConfig = None):
        self._config = config or self.get_default_config()

    def get_default_config(self):
        return StyleTransferConfig()

    def get_inference(self, content_image: Image, style_image: Image, config: StyleTransferConfig = None):
        return StyleTransferInference(config or self.config, content_image, style_image)

    @property
    def type(self):
        assert False, print("Should be overriden")

    @property
    def config(self):
        return deepcopy(self._config)
