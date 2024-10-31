# BSD 2-Clause License
#
# Copyright (c) 2024, Christoph Neuhauser
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import sys
import math
import itertools
import pathlib
import subprocess
from typing import Callable
import html
import smtplib
import ssl
from email.message import EmailMessage
from email.headerregistry import Address
from email.utils import formatdate
import skimage
import skimage.io
import skimage.metrics
import lpips
import torch
import torchvision
import matplotlib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def send_mail(
        sender_name, sender_email_address, user_name, password,
        recipient_name, recipient_email_address,
        subject, message_text_raw, message_text_html):
    if sender_email_address.endswith('@in.tum.de'):
        smtp_server = 'mail.in.tum.de'
        port = 587  # STARTTLS
    elif sender_email_address.endswith('@tum.de'):
        smtp_server = 'postout.lrz.de'
        port = 587  # STARTTLS
    else:
        raise Exception(f'Error: Unexpected provider in e-mail address {sender_email_address}!')

    context = ssl.create_default_context()
    server = smtplib.SMTP(smtp_server, port)
    server.ehlo()
    server.starttls(context=context)
    server.ehlo()
    server.login(user_name, password)

    message = EmailMessage()
    message['Subject'] = subject
    message['From'] = Address(display_name=sender_name, addr_spec=sender_email_address)
    message['To'] = Address(display_name=recipient_name, addr_spec=recipient_email_address)
    message['Date'] = formatdate(localtime=True)
    message.set_content(message_text_raw)
    message.add_alternative(message_text_html, subtype='html')

    server.sendmail(sender_email_address, recipient_email_address, message.as_string())

    server.quit()


def escape_html(s):
    s_list = html.escape(s, quote=False).splitlines(True)
    s_list_edit = []
    for se in s_list:
        se_notrail = se.lstrip()
        new_se = se_notrail
        for i in range(len(se) - len(se_notrail)):
            new_se = '&nbsp;' + new_se
        s_list_edit.append(new_se)
    s = ''.join(s_list_edit)
    return s.replace('\n', '<br/>\n')


scenes_dir = '/mnt/data/3DGS/360_v2'
train_dir = '/mnt/data/3DGS/train'
#scenes = [('bonsai', 'images_2')]
scenes = [('bicycle', 'images_4'), ('room', 'images_2')]
case = 'train'
iterations = 30000
img_idx = '00000'
scale_factors = [2]
configurations = [
    None,
    'ninasr_b1',
]


metrics = ['MSE', 'RMSE', 'PSNR', 'SSIM', 'LPIPS_Alex', 'LPIPS_VGG']
loss_fn_alex = lpips.LPIPS(net='alex')
loss_fn_vgg = lpips.LPIPS(net='vgg')


def skimage_to_torch(img):
    t = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    tensor = t(skimage.img_as_float(img)).float()
    tensor = tensor[None, 0:3, :, :] * 2 - 1
    return tensor


def compare_images(filename_gt, filename_approx):
    img_gt = skimage.io.imread(filename_gt)
    img_approx = skimage.io.imread(filename_approx)
    mse = skimage.metrics.mean_squared_error(img_gt, img_approx)
    psnr = skimage.metrics.peak_signal_noise_ratio(img_gt, img_approx)
    data_range = img_gt.max() - img_approx.min()
    ssim = skimage.metrics.structural_similarity(
        img_gt, img_approx, data_range=data_range, channel_axis=-1, multichannel=True)

    img0 = skimage_to_torch(img_gt)
    img1 = skimage_to_torch(img_approx)
    d_alex = loss_fn_alex(img0, img1).item()
    d_vgg = loss_fn_vgg(img0, img1).item()

    return {
        'MSE': mse,
        'RMSE': math.sqrt(mse),
        'PSNR': psnr,
        'SSIM': ssim,
        'LPIPS_Alex': d_alex,
        'LPIPS_VGG': d_vgg,
    }


def eval_train(eval_name, gt_image_dir, image_dirs):
    print(f'Starting test \'{eval_name}\'...')
    filename_gt = os.path.join(gt_image_dir, f'{img_idx}.png')
    results = []
    for test_dir in image_dirs:
        test_name = os.path.basename(test_dir)
        image_dir = os.path.join(test_dir, case, f'ours_{iterations}', 'renders')
        print(f"Test '{test_name}'...")
        filename_approx = os.path.join(image_dir, f'{img_idx}.png')
        result = compare_images(filename_gt, filename_approx)
        result['name'] = test_name
        results.append(result)

    results = sorted(results, key=lambda d: d['name'])
    print(results)
    #plot_results(base_dir, args.sf, results)


commands = []
for scene in scenes:
    scene_dir = os.path.join(scenes_dir, scene[0])
    images_dir = scene[1]

    # Check if ground truth was trained.
    gt_model_dir = os.path.join(train_dir, f'{scene[0]}_default')
    gt_image_dir = os.path.join(gt_model_dir, case, f'ours_{iterations}', 'renders')
    if not os.path.exists(gt_model_dir):
        commands.append([
            'python3', 'train.py',
            '-s', scene_dir, '-m', gt_model_dir, '--images', images_dir, '--antialiasing', '--eval',
            '--test_iterations', '7000', '15000', '30000'
        ])
    if not os.path.exists(gt_model_dir):
        commands.append(['python3', 'render.py', '-m', gt_model_dir, '--antialiasing'])

    for sf in scale_factors:
        image_dirs = []
        for configuration in configurations:
            config_name = configuration if configuration is not None else 'default'
            model_dir = os.path.join(train_dir, f'{scene[0]}_{config_name}')
            if not os.path.exists(model_dir):
                command_train = [
                    'python3', 'train.py',
                    '-s', scene_dir, '-m', model_dir, '--images', images_dir, '--antialiasing', '--eval',
                    '--test_iterations', '7000', '15000', '30000',
                    '--sf', str(sf)
                ]
                if configuration is not None:
                    command_train += [
                        '--sf', str(sf),
                        '--upscaling_method', 'torchsr',
                        '--upscaling_param', f'{configuration}'
                    ]
                else:
                    command_train += ['--downscale', str(sf)]
                commands.append(command_train)

            image_dir = os.path.join(model_dir, case, f'ours_{iterations}', 'renders')
            image_dirs.append(image_dir)
            if not os.path.exists(image_dir):
                commands.append(['python3', 'render.py', '-m', model_dir, '--antialiasing'])

        commands.append(lambda: eval_train(f'{scene[0]}_x{sf}', gt_image_dir, image_dirs))


if __name__ == '__main__':
    matplotlib.rcParams.update({'font.family': 'Linux Biolinum O'})
    matplotlib.rcParams.update({'font.size': 17.5})

    shall_send_email = True
    pwd_path = os.path.join(pathlib.Path.home(), 'Documents', 'mailpwd.txt')
    use_email = pathlib.Path(pwd_path).is_file()
    if use_email:
        with open(pwd_path, 'r') as file:
            lines = [line.rstrip() for line in file]
            sender_name = lines[0]
            sender_email_address = lines[1]
            user_name = lines[2]
            password = lines[3]
            recipient_name = lines[4]
            recipient_email_address = lines[5]

    for command in commands:
        if isinstance(command, Callable):
            command()
            continue

        print(f"Running '{' '.join(command)}'...")
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (output, err) = proc.communicate()
        proc_status = proc.wait()
        if proc_status != 0:
            stderr_string = err.decode('utf-8')
            stdout_string = output.decode('utf-8')

            if use_email:
                message_text_raw = f'The following command failed with code {proc_status}:\n'
                message_text_raw += ' '.join(command) + '\n\n'
                message_text_raw += '--- Output from stderr ---\n'
                message_text_raw += stderr_string
                message_text_raw += '---\n\n'
                message_text_raw += '--- Output from stdout ---\n'
                message_text_raw += stdout_string
                message_text_raw += '---'

                message_text_html = \
                    '<html>\n<head><meta http-equiv="Content-Type" content="text/html; charset=utf-8"></head>\n<body>\n'
                message_text_html += f'The following command failed with code {proc_status}:<br/>\n'
                message_text_html += ' '.join(command) + '<br/><br/>\n\n'
                message_text_html += '<font color="red" style="font-family: \'Courier New\', monospace;">\n'
                message_text_html += '--- Output from stderr ---<br/>\n'
                message_text_html += escape_html(stderr_string)
                message_text_html += '---</font>\n<br/><br/>\n\n'
                message_text_html += '<font style="font-family: \'Courier New\', monospace;">\n'
                message_text_html += '--- Output from stdout ---<br/>\n'
                message_text_html += escape_html(stdout_string)
                message_text_html += '---</font>\n'
                message_text_html += '</body>\n</html>'

                if shall_send_email:
                    send_mail(
                        sender_name, sender_email_address, user_name, password,
                        recipient_name, recipient_email_address,
                        'Error while generating images', message_text_raw, message_text_html)

            print('--- Output from stdout ---')
            print(stdout_string.rstrip('\n'))
            print('---\n')
            print('--- Output from stderr ---', file=sys.stderr)
            print(stderr_string.rstrip('\n'), file=sys.stderr)
            print('---', file=sys.stderr)
            sys.exit(1)
            #raise Exception(f'Process returned error code {proc_status}.')
        elif not shall_send_email:
            stderr_string = err.decode('utf-8')
            stdout_string = output.decode('utf-8')
            print('--- Output from stdout ---')
            print(stdout_string.rstrip('\n'))
            print('---\n')
            print('--- Output from stderr ---', file=sys.stderr)
            print(stderr_string.rstrip('\n'), file=sys.stderr)
            print('---', file=sys.stderr)

    message_text_raw = 'run.py finished successfully'
    message_text_html = \
        '<html>\n<head><meta http-equiv="Content-Type" content="text/html; charset=utf-8"></head>\n<body>\n'
    message_text_html += 'run.py finished successfully'
    message_text_html += '</body>\n</html>'
    if shall_send_email:
        send_mail(
            sender_name, sender_email_address, user_name, password,
            recipient_name, recipient_email_address,
            'run.py finished successfully', message_text_raw, message_text_html)
    print('Finished.')
