# https://docs.pytest.org/en/latest/

# !!! Don't forget to install `pytest`: python -m pip install pytest


from PIL import Image
from styletransfer.styletransfer import StyleTransferInference, StyleTransferConfig


class Test__StyleTransferInference:

    def make_image(self, width, height):
        return Image.new('RGB', (width, height), None)

    def test__resize_image(self):
        image = self.make_image(3, 2)
        ref = self.make_image(2, 3)
        inference = StyleTransferInference(None, None, None)
        resized = inference._resize_image(image, ref)

        # TEST A resized image should be the same size as a ref image
        assert resized.size == ref.size

    def test__prepare_images(self):
        content = self.make_image(5, 6)
        style = self.make_image(4, 3)
        config = StyleTransferConfig(image_size=3)
        inference = StyleTransferInference(config, content, style)
        inference._prepare_images()

        # TEST Prepared images should be the same sizes when input images are different
        assert inference._content_image.size == inference._style_image.size

        # TEST Width and height of prepared images should be not greater than config.image_size
        assert inference._content_image.size[0] <= config.image_size and \
               inference._content_image.size[1] <= config.image_size
