import argparse
import cv2
import glob
import numpy as np
import time
import os
import torch
from pathlib import Path
from basicsr.utils import imwrite
from datetime import datetime
from flask import Flask, jsonify, request, send_file
from gfpgan import GFPGANer


class Server:
    def __init__(self):
        self.file_index = 0
        self.restorer = None

    def make_filename(self):
        self.file_index += 1
        now = datetime.now()
        formatted_time = now.strftime('%Y%m%d%H%M%S%f')[:-3] + "_" + str(self.file_index)
        return formatted_time

    def init_model(self, version, upscale, bg_upsampler, bg_tile):
        if bg_upsampler == 'realesrgan':
            if not torch.cuda.is_available():  # CPU
                import warnings
                warnings.warn('The unoptimized RealESRGAN is slow on CPU. We do not use it. '
                              'If you really want to use it, please modify the corresponding codes.')
                bg_upsampler = None
            else:
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from realesrgan import RealESRGANer
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
                bg_upsampler = RealESRGANer(
                    scale=2,
                    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                    model=model,
                    tile=bg_tile,
                    tile_pad=10,
                    pre_pad=0,
                    half=True)  # need to set False in CPU mode
        else:
            bg_upsampler = None

        if version == '1':
            arch = 'original'
            channel_multiplier = 1
            model_name = 'GFPGANv1'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth'
        elif version == '1.2':
            arch = 'clean'
            channel_multiplier = 2
            model_name = 'GFPGANCleanv1-NoCE-C2'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth'
        elif version == '1.3':
            arch = 'clean'
            channel_multiplier = 2
            model_name = 'GFPGANv1.3'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'
        elif version == '1.4':
            arch = 'clean'
            channel_multiplier = 2
            model_name = 'GFPGANv1.4'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
        elif version == 'RestoreFormer':
            arch = 'RestoreFormer'
            channel_multiplier = 2
            model_name = 'RestoreFormer'
            url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth'
        else:
            raise ValueError(f'Wrong model version {version}.')

        # determine model paths
        model_path = os.path.join('experiments/pretrained_models', model_name + '.pth')
        if not os.path.isfile(model_path):
            model_path = os.path.join('gfpgan/weights', model_name + '.pth')
        if not os.path.isfile(model_path):
            # download pre-trained models from url
            model_path = url

        self.restorer = GFPGANer(
            model_path=model_path,
            upscale=upscale,
            arch=arch,
            channel_multiplier=channel_multiplier,
            bg_upsampler=bg_upsampler)

    def enhance_image(self, img_path):
        only_center_face = False
        aligned = False
        bg_tile = 400
        weight = 0.5
        input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # restore faces and background if necessary
        cropped_faces, restored_faces, restored_img = self.restorer.enhance(
            input_img,
            has_aligned=aligned,
            only_center_face=only_center_face,
            paste_back=True,
            weight=weight)
        return restored_img


server = Server()
server.init_model("1.4", 2, "realesrgan", 400)
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)


@app.route('/health')
def health():
    return jsonify(code=0, msg=f'I am OK')


@app.route('/enhance', methods=['GET', 'POST'])
def get_enhance_image():
    time_start = time.time()
    image_input = request.files["image"]
    image_name = server.make_filename()
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)
    image_input.save(image_path)
    image_out = server.enhance_image(image_path)
    image_out_path = image_path + "_out.jpg"
    imwrite(image_out, image_out_path)
    time_end = time.time()
    print(f'enhance {image_path} cost:{time_end-time_start:.3f}s')
    # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return send_file(image_out_path, mimetype='image/jpeg')


if __name__ == '__main__':
    port = 9911
    print(f"server started at port:{port}")
    app.run(port = port, debug=True)
