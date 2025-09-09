import torch
import torch.nn as nn
import timm
from clft.reassemble import Reassemble
from clft.fusion import Fusion
from clft.head import HeadSeg, HeadDepth
import numpy as np


class ViTSegmentation(nn.Module):
    def __init__(self, mode, num_classes=5):
        super().__init__()
        
        # Hardcoded config values
        RGB_tensor_size = (3, 384, 384)
        XYZ_tensor_size = (3, 384, 384)
        patch_size = 16
        emb_dim = 768
        resample_dim = 256
        read = 'projection'
        self.hooks = [2, 5, 8, 11]
        reassemble_s = [4, 8, 16, 32]
        nclasses = num_classes
        type_ = 'segmentation'
        model_timm = 'vit_base_patch16_384'
        
        self.mode = mode
        self.type_ = type_
        
        # Transformer encoder
        self.transformer_encoders = timm.create_model(model_timm, pretrained=True)
        
        # Register hooks
        self.activation = {}
        self._get_layers_from_hooks(self.hooks)
        
        # Reassembles and Fusions
        self.reassembles_RGB = nn.ModuleList([
            Reassemble(RGB_tensor_size, read, patch_size, s, emb_dim, resample_dim) for s in reassemble_s
        ])
        self.reassembles_XYZ = nn.ModuleList([
            Reassemble(XYZ_tensor_size, read, patch_size, s, emb_dim, resample_dim) for s in reassemble_s
        ])
        self.fusions = nn.ModuleList([Fusion(resample_dim) for _ in reassemble_s])
        
        # Head
        if type_ == "full":
            self.head_depth = HeadDepth(resample_dim)
            self.head_segmentation = HeadSeg(resample_dim, nclasses)
        elif type_ == "depth":
            self.head_depth = HeadDepth(resample_dim)
            self.head_segmentation = None
        else:
            self.head_depth = None
            self.head_segmentation = HeadSeg(resample_dim, nclasses)

    def _get_layers_from_hooks(self, hooks):
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output
            return hook
        for h in hooks:
            self.transformer_encoders.blocks[h].register_forward_hook(get_activation('t' + str(h)))

    def forward(self, rgb, lidar, modal=None):
        if modal is None:
            modal = self.mode
        
        # Apply transformer to lidar
        t = self.transformer_encoders(lidar)
        
        previous_stage = None
        for i in np.arange(len(self.fusions) - 1, -1, -1):
            hook_to_take = 't' + str(self.hooks[i])
            activation_result = self.activation[hook_to_take]
            
            if modal == 'rgb':
                reassemble_result_RGB = self.reassembles_RGB[i](activation_result)
                reassemble_result_XYZ = torch.zeros_like(reassemble_result_RGB)
            elif modal == 'lidar':
                reassemble_result_XYZ = self.reassembles_XYZ[i](activation_result)
                reassemble_result_RGB = torch.zeros_like(reassemble_result_XYZ)
            elif modal == 'cross_fusion':
                reassemble_result_RGB = self.reassembles_RGB[i](activation_result)
                reassemble_result_XYZ = self.reassembles_XYZ[i](activation_result)
            
            fusion_result = self.fusions[i](reassemble_result_RGB, reassemble_result_XYZ, previous_stage, modal)
            previous_stage = fusion_result
        
        out_depth = None
        out_segmentation = None
        if self.head_depth is not None:
            out_depth = self.head_depth(previous_stage)
        if self.head_segmentation is not None:
            out_segmentation = self.head_segmentation(previous_stage)
        
        if self.type_ == "full":
            return out_depth, out_segmentation
        elif self.type_ == "depth":
            return out_depth
        else:
            return out_segmentation


