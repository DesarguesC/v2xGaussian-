import torch, pdb
import torch.nn as nn
import torch.nn.functional as F
from unidepth.models import UniDepthV1
from einops import rearrange
from flash3d.models.encoder.resnet_encoder import ResnetEncoder
from flash3d.models.decoder.resnet_decoder import ResnetDecoder, ResnetDepthDecoder

class UniDepthExtended(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # self.unidepth = UniDepthV1.from_pretrained("../lpiccinelli-eth/unidepth", local_files_only=True)
        self.unidepth = UniDepthV1.from_pretrained("../lpiccinelli/unidepth-v1-vitl14", local_files_only=True)

        self.parameters_to_train = []
        if cfg.model.backbone.name == "resnet":
            self.encoder = ResnetEncoder(
                num_layers=cfg.model.backbone.num_layers,
                pretrained=cfg.model.backbone.weights_init == "pretrained",
                bn_order=cfg.model.backbone.resnet_bn_order,
            )
            # change encoder to take depth as conditioning
            if cfg.model.backbone.depth_cond:
                self.encoder.encoder.conv1 = nn.Conv2d(
                    4,
                    self.encoder.encoder.conv1.out_channels,
                    kernel_size = self.encoder.encoder.conv1.kernel_size,
                    padding = self.encoder.encoder.conv1.padding,
                    stride = self.encoder.encoder.conv1.stride
                )
            self.parameters_to_train += [{"params": self.encoder.parameters()}]
            models = {}
            if cfg.model.gaussians_per_pixel > 1:
                models["depth"] = ResnetDepthDecoder(cfg=cfg, num_ch_enc=self.encoder.num_ch_enc)
                self.parameters_to_train +=[{"params": models["depth"].parameters()}]
            for i in range(cfg.model.gaussians_per_pixel):
                models["gauss_decoder_"+str(i)] = ResnetDecoder(cfg=cfg,num_ch_enc=self.encoder.num_ch_enc)
                self.parameters_to_train += [{"params": models["gauss_decoder_"+str(i)].parameters()}]
                if cfg.model.one_gauss_decoder:
                    break
            self.models = nn.ModuleDict(models)

    def get_parameter_groups(self):
        # only the resnet encoder and gaussian parameter decoder are optimisable
        return self.parameters_to_train

    def forward(self, inputs):
        # prediting the depth for the first layer with pre-trained depth
        if ('unidepth', 0, 0) in inputs.keys() and inputs[('unidepth', 0, 0)] is not None:
            depth_outs = dict()
            depth_outs["depth"] = inputs[('unidepth', 0, 0)]
        else:
            with torch.no_grad():
                intrinsics = inputs[("K_src", 0)] if ("K_src", 0) in inputs.keys() else None
                depth_outs = self.unidepth.infer(inputs["color_aug", 0, 0], intrinsics=intrinsics)
        outputs_gauss = {}

        outputs_gauss[("K_src", 0)] = inputs[("K_src", 0)] if ("K_src", 0) in inputs.keys() else depth_outs["intrinsics"]
        outputs_gauss[("inv_K_src", 0)] = torch.linalg.inv(outputs_gauss[("K_src", 0)])

        # pdb.set_trace()
        if self.cfg.model.backbone.depth_cond:
            # division by 20 is to put depth in a similar range to RGB
            if(len(inputs["color_aug", 0, 0].shape) < len(depth_outs["depth"].shape)):
                inputs["color_aug", 0, 0] = inputs["color_aug", 0, 0][None,:,:,:]
            input = torch.cat([inputs["color_aug", 0, 0], depth_outs["depth"] / 20.0], dim=1)
            # TODO: check why: [3, 320, 576] ~ [1, 1, 320, 576] | 保存下看看
        else:
            input = inputs["color_aug", 0, 0]

        # input.shape: [1, 4, 320, 576]
        encoded_features = self.encoder(input)
        # predict multiple gaussian depths
        if self.cfg.model.gaussians_per_pixel > 1:
            depth = self.models["depth"](encoded_features)
            depth[("depth", 0)] = rearrange(depth[("depth", 0)], "(b n) ... -> b n ...", n=self.cfg.model.gaussians_per_pixel - 1)
            depth[("depth", 0)] = torch.cumsum(torch.cat((depth_outs["depth"][:,None,...], depth[("depth", 0)]), dim=1), dim=1)
            outputs_gauss[("depth", 0)] = rearrange(depth[("depth", 0)], "b n c ... -> (b n) c ...", n = self.cfg.model.gaussians_per_pixel)
        else:
            outputs_gauss[("depth", 0)] = depth_outs["depth"]
        # predict multiple gaussian parameters
        gauss_outs = dict()
        for i in range(self.cfg.model.gaussians_per_pixel):
            outs = self.models["gauss_decoder_"+str(i)](encoded_features)
            # keys: ['gauss_opacity', 'gauss_scaling', 'gauss_rotation', 'gauss_features_dc', 'gauss_offset', 'gauss_features_rest']
            if self.cfg.model.one_gauss_decoder:
                gauss_outs |= outs
                break
            else:
                for key, v in outs.items():
                    gauss_outs[key] = outs[key] if i==0 else torch.cat([gauss_outs[key], outs[key]], dim=1)
        # keys: ['gauss_opacity', 'gauss_scaling', 'gauss_rotation', 'gauss_features_dc', 'gauss_offset', 'gauss_features_rest']
        for key, v in gauss_outs.items():
            gauss_outs[key] = rearrange(gauss_outs[key], 'b n ... -> (b n) ...')
        outputs_gauss |= gauss_outs

        for (k,v) in outputs_gauss.items(): # TODO: check type - float32 ? float64 ?
            outputs_gauss[k] = v.to(torch.float32)

        return outputs_gauss