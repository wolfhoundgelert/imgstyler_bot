# https://docs.pytest.org/en/latest/

from PIL import Image
from styletransfer.styletransfer import StyleTransferInference, StyleTransferConfig

def make_inference():
    content = Image.open('../pic/img_2.png')
    style = Image.open('../pic/img_3.png')
    assert content.size != style.size, print("Test images should be different sizes")
    return StyleTransferInference(StyleTransferConfig(), content, style)


# A resized image should be the same size as a ref image
def test_StyleTransferInference_resize_image():
    image = Image.open('../pic/img_2.png')
    ref = Image.open('../pic/img_3.png')
    inference = make_inference()
    resized = inference._resize_image(image, ref)
    assert resized.size == ref.size


# Resized images should be the same sizes when input images are different
def test_StyleTransferInference_prepare_images():
    inf = make_inference()
    inf._prepare_images()
    assert inf._content_image.size == inf._style_image.size
