import os
from collections import OrderedDict

import torch

from ModelZoo.NN.edsr import EDSR

MODEL_DIR = '/data/zyli/projects/LAM_Demo/ModelZoo/models/'


NN_LIST = [
    'RCAN',
    'CARN',
    'RRDBNet',
    'RNAN', 
    'SAN',
    'PAN',
    'HAN',
    'IGNN',
    'SwinIR',
    'EDSR'
]


MODEL_LIST = {
    'RCAN': {
        'Base': 'RCAN.pt',
    },
    'CARN': {
        'Base': 'CARN_7400.pth',
    },
    'RRDBN': {
        'Base': 'RRDBNet_PSNR_SRx4_DF2K_official-150ff491.pth',
    },
    'SAN': {
        'Base': 'SAN_BI4X.pt',
    },
    'RNAN': {
        'Base': 'RNAN_SR_F64G10P48BIX4.pt',
    },
    'PAN': {
        'Base': 'PANx4_DF2K.pth',
    },
    'HAN': {
        'Base': 'HAN_BIX4.pt',
    },
    'IGNN': {
        'Base': 'IGNN_x4.pth',
    },
    'SwinIR': {
        'Base': '001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth',
    },
    'EDSR': {
        'Base': 'EDSR-64-16_15000.pth'
    }
}

def print_network(model, model_name):
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('Network [%s] was created. Total number of parameters: %.1f kelo. '
          'To see the architecture, do print(network).'
          % (model_name, num_params / 1000))


def get_model(model_name, factor=4, num_channels=3):
    """
    All the models are defaulted to be X4 models, the Channels is defaulted to be RGB 3 channels.
    :param model_name:
    :param factor:
    :param num_channels:
    :return:
    """
    print(f'Getting SR Network {model_name}')
    if model_name.split('-')[0] in NN_LIST:

        if model_name == 'RCAN':
            from .NN.rcan import RCAN
            net = RCAN(factor=factor, num_channels=num_channels)

        elif model_name == 'EDSR':
            from .NN.edsr import EDSR
            net = EDSR()

        elif model_name == 'CARN':
            from .CARN.carn import CARNet
            net = CARNet(factor=factor, num_channels=num_channels)

        elif model_name == 'RRDBNet':
            from .NN.rrdbnet import RRDBNet
            net = RRDBNet(num_in_ch=num_channels, num_out_ch=num_channels)

        elif model_name == 'SAN':
            from .NN.san import SAN
            net = SAN(factor=factor, num_channels=num_channels)

        elif model_name == 'RNAN':
            from .NN.rnan import RNAN
            net = RNAN(factor=factor, num_channels=num_channels)

        elif model_name == 'PAN':
            from .NN.pan_arch import PAN
            net = PAN(num_in_ch=3, num_out_ch=3, num_feat=40, unf=24, num_block=16, upscale=4)

        elif model_name == 'HAN':
            from .NN.han_arch import HAN
            net = HAN(factor=4, num_channels=3)

        elif model_name == 'IGNN':
            from .NN.ignn_arch import IGNN
            net = IGNN()

        elif model_name == 'SwinIR':
            from .NN.swinir_arch import SwinIR
            net = SwinIR(upscale=4, in_chans=3, img_size=48, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')

        else:
            raise NotImplementedError()

        print_network(net, model_name)
        return net
    else:
        raise NotImplementedError()


def load_model(model_loading_name):
    """
    :param model_loading_name: model_name-training_name
    :return:
    """
    splitting = model_loading_name.split('@')
    if len(splitting) == 1:
        model_name = splitting[0]
        training_name = 'Base'
    elif len(splitting) == 2:
        model_name = splitting[0]
        training_name = splitting[1]
    else:
        raise NotImplementedError()
    assert model_name in NN_LIST or model_name in MODEL_LIST.keys(), 'check your model name before @'
    net = get_model(model_name)
    state_dict_path = os.path.join(MODEL_DIR, MODEL_LIST[model_name][training_name])
    print(f'Loading model {state_dict_path} for {model_name} network.')
    if model_name ==  'IGNN':
        checkpoint = torch.load(state_dict_path)
        net_state_dict = OrderedDict()
        for k, v in checkpoint['net_state_dict'].items():
            name = k[7:]
            net_state_dict[name] = v
        net.load_state_dict(net_state_dict, False)
        return net
    elif model_name == 'SwinIR':
        loadnet = torch.load(state_dict_path)
        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'
        net.load_state_dict(loadnet[keyname], strict=True)
    else:
        state_dict = torch.load(state_dict_path, map_location='cpu')
        net.load_state_dict(state_dict)
    return net




