import logging
import threading
import multiprocessing
from os import getpid
from telegram import Update
from telegram.ext import filters, MessageHandler, ApplicationBuilder, ContextTypes
from io import BytesIO
from PIL import Image
import requests

from styletransfer.styletransfer import StyleTransferType, StyleTransfer, StyleTransferInference, StyleTransferConfig


class InferenceWorker:

    @staticmethod
    def get_worker(style_transfer: StyleTransfer, pid_to_result,
                   content_image: Image, style_image: Image, config: StyleTransferConfig = None):

        inference = style_transfer.get_inference(content_image, style_image, config)
        worker = InferenceWorker(pid_to_result, inference)
        return worker

    def __init__(self, pid_to_result, inference: StyleTransferInference):
        self._pid_to_result = pid_to_result
        self._inference = inference

    # Defining __call__ method
    def __call__(self):
        self._pid_to_result[getpid()] = self._inference()


class Application:

    def __init__(self, style_transfer_type: StyleTransferType):

        # style transfer part:
        self._style_transfer_type = style_transfer_type
        self._user_id_to_first_image = {}  # dict for saving the first photo from a user (we need 2 photos)
        self._pid_to_result = multiprocessing.Manager().dict()  # dict for saving results via multiprocessing
        self._pid_to_chat_id = {}  # dict for saving chat id via multiprocessing for sending results back to users
        self._timer = None  # check results by timer via multiprocessing
        self._style_transfer = self._build_style_transfer()

        # telegram bot part:
        with open('../token.txt') as file:
            self._token = file.read()
        self._bot = self._build_and_run_bot()

    def _build_style_transfer(self):
        if self._style_transfer_type == StyleTransferType.Gatys:
            from styletransfer.gatys.gatys import Gatys
            return Gatys()

        if self._style_transfer_type == StyleTransferType.Magenta:
            from styletransfer.magenta.magenta import Magenta
            return Magenta()

        if self._style_transfer_type == StyleTransferType.MSGNet \
                or self._style_transfer_type == StyleTransferType.MSGNetCustomTrain:

            from styletransfer.msgnet.msgnet import MSGNet, MSGNetConfig

            model_path = './styletransfer/msgnet/'
            model_path += '21styles.model' if self._style_transfer_type == StyleTransferType.MSGNet \
                                            else 'my9styles.model'

            config = MSGNetConfig(model_path=model_path)

            return MSGNet(config)

    def _build_and_run_bot(self):
        bot = ApplicationBuilder().token(self._token).build()

        image_handler = MessageHandler(filters.PHOTO | filters.Document.IMAGE, self._on_image)
        bot.add_handler(image_handler)

        # Should be the last handler:
        all_handler = MessageHandler(filters.ALL, self._on_start)
        bot.add_handler(all_handler)

        # ATTENTION Don't add more handlers here

        bot.run_polling()
        return bot

    def _check_results(self):
        for pid in self._pid_to_result.keys():
            result = self._pid_to_result.pop(pid)
            chat_id = self._pid_to_chat_id.pop(pid)
            self._send_image_to_chat(result, chat_id)

        if len(self._pid_to_result.keys()):
            self._check_results()
        elif len(self._pid_to_chat_id.keys()):
            self._start_timer()

    def _start_timer(self):
        self._timer = threading.Timer(1, self._check_results)
        self._timer.start()

    def _send_image_to_chat(self, result, chat_id):
        bio = BytesIO()
        result.save(bio, 'JPEG')
        bio.seek(0)

        api_url = f'https://api.telegram.org/bot{self._token}/sendPhoto?chat_id={chat_id}'
        files = {"photo": bio}

        try:
            response = requests.post(api_url, files=files)
            print(f"Response: {response.text}")
        except Exception as e:
            print(f"Exception: {e}")

    async def _on_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        txt = ("Hi! I'm Image Styler. Send me 2 photos. I'll take the content from the first one and the style from "
               "the second one. Then I'll generate a new image with the taken content in the taken style and send it"
               " back.")
        await context.bot.send_message(chat_id=update.effective_chat.id, text=txt)

    async def _on_image(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg = update.message
        bot = context.bot

        file_id = msg.photo[-1].file_id if msg.photo else msg.document.file_id
        file = await bot.get_file(file_id)
        out = BytesIO()
        await file.download_to_memory(out)
        img = Image.open(out)

        chat_id = msg.chat_id
        if chat_id not in self._user_id_to_first_image.keys():
            # TODO Doesn't look good if user sends both images at once:
            # txt = "...please send another one for style..."
            # await context.bot.send_message(chat_id=update.effective_chat.id, text=txt)

            self._user_id_to_first_image[chat_id] = img
            print(f"The first image from user {chat_id} has been saved, waiting for the second one")
        else:
            if self._style_transfer_type == StyleTransferType.Gatys:  # the slow one
                txt = "...in progress. Please wait, it may take a few minutes..."
            else:
                txt = "...in progress. Please wait, it may take for a while..."

            await context.bot.send_message(chat_id=update.effective_chat.id, text=txt)

            print(f"The second image from user {chat_id} has been received and we start the style transfer")
            content_image = self._user_id_to_first_image.pop(chat_id)
            style_image = img

            inference_worker = InferenceWorker.get_worker(
                self._style_transfer, self._pid_to_result, content_image, style_image)

            p = multiprocessing.Process(target=inference_worker)
            p.start()
            self._pid_to_chat_id[p.pid] = chat_id

            self._start_timer()


if __name__ == '__main__':

    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    # Choose a stile transfer here:
    # TODO make an external config and update the readme
    style_transfer_type = StyleTransferType.MSGNet

    Application(style_transfer_type)
