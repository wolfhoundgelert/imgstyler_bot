# Source: https://arxiv.org/abs/1508.06576
# Source: https://pytorch.org/tutorials/advanced/neural_style_tutorial.html

from torch import mm
from torch import tensor
from torch import no_grad
from torch.nn import Module, Sequential, Conv2d, ReLU, MaxPool2d
from torch.nn.functional import mse_loss
from torch.optim import LBFGS
from torchvision.transforms import ToPILImage, ToTensor
from torchvision.models.vgg import vgg19, VGG19_Weights

from PIL import Image
from tqdm import tqdm

from styletransfer.styletransfer import \
    StyleTransfer, StyleTransferType, StyleTransferConfig, StyleTransferInference
# from styletransfer.torchdevice import device


class GatysConfig(StyleTransferConfig):

    def __init__(self,
                 image_size: int = 256,
                 keep_content_aspect_ratio: bool = True,
                 loss_convergence_threshold=20,
                 num_steps=300,
                 content_layers=['conv_4'],
                 style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5'],
                 style_weight=100000,
                 content_weight=1
                 ):

        super().__init__(image_size, keep_content_aspect_ratio)
        self.loss_convergence_threshold = loss_convergence_threshold
        self.num_steps = num_steps
        # desired depth layers to compute style/content losses :
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.style_weight = style_weight
        self.content_weight = content_weight


class ContentLoss(Module):

    def __init__(self, target):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = mse_loss(input, self.target)
        return input


class StyleLoss(Module):

    @staticmethod
    def gram_matrix(target):
        a, b, c, d = target.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = target.view(a * b, c * d)  # resize F_XL into \hat F_XL

        G = mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

    def __init__(self, target):
        super().__init__()
        self.target = StyleLoss.gram_matrix(target).detach()

    def forward(self, input):
        G = StyleLoss.gram_matrix(input)
        self.loss = mse_loss(G, self.target)
        return input


# create a module to normalize input image so we can easily put it in a nn.Sequential
class Normalization(Module):

    cnn_normalization_mean = tensor([0.485, 0.456, 0.406])  # .to(device)
    cnn_normalization_std = tensor([0.229, 0.224, 0.225])  # .to(device)

    def __init__(self):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = Normalization.cnn_normalization_mean.clone().detach().view(-1, 1, 1)
        self.std = Normalization.cnn_normalization_std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

class GatysInference(StyleTransferInference):

    @staticmethod
    def _image_to_tensor(image: Image):
        tnsr = GatysInference._image_to_tensor_transform(image).unsqueeze(0)  # .to(device)
        return tnsr

    @staticmethod
    def _tensor_to_image(tnsr):
        trs = ToPILImage()
        tnsr = tnsr.cpu().clone()  # We clone the tensor to not do changes on it
        img = trs(tnsr.squeeze(0))  # Remove the fake batch dimension, then transform
        return img

    _image_to_tensor_transform = ToTensor()

    _vgg19 = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()  # .to(device).eval()

    def __init__(self, config: GatysConfig, content_image: Image, style_image: Image):
        super().__init__(config, content_image, style_image)
        self._config = config  # resolve self._config as GatysConfig (not as StyleTransferConfig)

        # TODO refactor without self. variables (make computation methods static, keep only inheritance)
        self._content = None
        self._style = None
        self._input = None
        self._model = None
        self._content_losses = None
        self._style_losses = None
        self._output = None
        self._result_image = None

    def _prepare_images(self):
        super()._prepare_images()
        self._content = GatysInference._image_to_tensor(self._content_image)
        self._style = GatysInference._image_to_tensor(self._style_image)
        self._input = self._content.clone()

    def _prepare_style_model_and_losses(self):
        print('Building the style transfer model..')
        cnn = GatysInference._vgg19

        # normalization module
        normalization = Normalization()  # .to(device)

        # just in order to have iterable access to or list of content/style
        # losses
        content_losses = []
        style_losses = []

        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        model = Sequential(normalization)

        i = 0  # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                layer = ReLU(inplace=False)
            elif isinstance(layer, MaxPool2d):
                name = 'pool_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in self._config.content_layers:
                # add content loss:
                target = model(self._content).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in self._config.style_layers:
                # add style loss:
                target_feature = model(self._style).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        # now we trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        self._model = model
        self._content_losses = content_losses
        self._style_losses = style_losses

    def _get_input_optimizer(self):
        # this line to show that input is a parameter that requires a gradient
        optimizer = LBFGS([self._input], max_iter=1)
        return optimizer

    def _run_style_transfer(self):
        """Run the style transfer."""

        # We want to optimize the input and not the model parameters so we
        # update all the requires_grad fields accordingly
        self._input.requires_grad_(True)
        self._model.requires_grad_(False)

        optimizer = self._get_input_optimizer()

        print('Optimizing..')
        state = {'finished': False, 'style_loss': None, 'content_loss': None}

        for step in (pbar := tqdm(range(self._config.num_steps), position=0, leave=True)):
            if state['finished']:
                break

            def closure():
                # correct the values of updated input image
                with no_grad():
                    self._input.clamp_(0, 1)

                optimizer.zero_grad()
                self._model(self._input)
                style_score = 0
                content_score = 0

                for sl in self._style_losses:
                    style_score += sl.loss
                for cl in self._content_losses:
                    content_score += cl.loss

                style_score *= self._config.style_weight
                content_score *= self._config.content_weight

                loss = style_score + content_score
                loss.backward()

                txt = (f" Style Loss: {style_score.item():.2f}, "
                       f"Content Loss: {content_score.item():.2f}")
                pbar.set_description(txt, refresh=True)

                if loss.item() < self._config.loss_convergence_threshold:
                    state['finished'] = True  # End cycle
                    state['style_loss'] = style_score.item()
                    state['content_loss'] = content_score.item()

                return loss

            optimizer.step(closure)

        print((f"Finished as riched loss convergence threshold "
               f"({self._config.loss_convergence_threshold}) or number of steps ({self._config.num_steps})\n"))

        # a last correction...
        with no_grad():
            self._input.clamp_(0, 1)

    def _inference(self) -> Image:
        self._prepare_images()
        self._prepare_style_model_and_losses()
        self._run_style_transfer()
        self._result_image = GatysInference._tensor_to_image(self._input)
        return self._result_image

class Gatys(StyleTransfer):

    def __init__(self, config: GatysConfig = None):
        super().__init__(config)

    def get_default_config(self):
        return GatysConfig()

    def get_inference(self, content_image: Image, style_image: Image, config: GatysConfig = None):
        inference = GatysInference(config or self.config, content_image, style_image)
        return inference

    def type(self):
        return StyleTransferType.Gatys
