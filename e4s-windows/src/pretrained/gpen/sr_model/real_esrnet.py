import os
import torch
import numpy as np
from torch.nn import functional as F

from src.pretrained.gpen.sr_model.rrdbnet_arch import RRDBNet

class RealESRNet(object):
    def __init__(self, base_dir='./', model=None, scale=2, device='cuda'):
        self.base_dir = base_dir
        self.scale = scale
        self.device = device
        self.load_srmodel(base_dir, model)

    @staticmethod
    def _unwrap_state_dict(loadnet):
        if isinstance(loadnet, dict) and 'params_ema' in loadnet:
            return loadnet['params_ema']
        if isinstance(loadnet, dict) and 'params' in loadnet:
            return loadnet['params']
        return loadnet

    def load_srmodel(self, base_dir, model):
        if model is None:
            loadnet = torch.load(os.path.join(self.base_dir, 'weights', 'realesrnet_x2.pth'))
        else:
            loadnet = torch.load(os.path.join(self.base_dir, 'weights', model+'_x%d.pth'%self.scale))
        state_dict = self._unwrap_state_dict(loadnet)
        num_feat = state_dict['conv_first.weight'].shape[0]
        num_grow_ch = state_dict['body.0.rdb1.conv1.weight'].shape[0]
        num_block = len({
            key.split('.')[1]
            for key in state_dict.keys()
            if key.startswith('body.') and key.split('.')[1].isdigit()
        })
        self.srmodel = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=num_feat,
            num_block=num_block,
            num_grow_ch=num_grow_ch,
            scale=self.scale,
        )
        self.srmodel.load_state_dict(state_dict, strict=True)
        self.srmodel.eval()
        self.srmodel = self.srmodel.to(self.device)

    def process(self, img):
        img = img.astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).to(self.device)

        if self.scale == 2:
            mod_scale = 2
        elif self.scale == 1:
            mod_scale = 4
        else:
            mod_scale = None
        if mod_scale is not None:
            h_pad, w_pad = 0, 0
            _, _, h, w = img.size()
            if (h % mod_scale != 0):
                h_pad = (mod_scale - h % mod_scale)
            if (w % mod_scale != 0):
                w_pad = (mod_scale - w % mod_scale)
            img = F.pad(img, (0, w_pad, 0, h_pad), 'reflect')

        try:
            with torch.no_grad():
                output = self.srmodel(img)
            # remove extra pad
            if mod_scale is not None:
                _, _, h, w = output.size()
                output = output[:, :, 0:h - h_pad, 0:w - w_pad]
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)

            return output
        except Exception as e:
            print('sr failed:', e)
            return None
