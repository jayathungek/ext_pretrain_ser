import math
import copy
from pathlib import Path
from typing import Optional, Callable, Tuple
import collections
from itertools import repeat
from tqdm import tqdm
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from extpt.constants import *
    from extpt.helpers import AverageMeter, ConfusionMatrixMeter
    from extpt.transformer import EncoderLayer
    from extpt.open_clip.transformer import LayerNorm, Transformer, _expand_token
    from extpt.open_clip.model import build_vision_tower, CLIPVisionCfg
    from extpt.jepa.vision_transformer import load_pretrained_videovit
    from extpt.resnet.resnet import ResNet50
    from extpt.face import get_roi_presets, VideoTransformer
    from extpt.astfm.transformer import ASTModel
    from extpt.astfm.emo_ast import EmoAST
    from extpt.helpers import get_best_checkpoint
except ModuleNotFoundError:
    from constants import *
    from helpers import AverageMeter, ConfusionMatrixMeter
    from transformer import EncoderLayer
    from open_clip.transformer import LayerNorm, Transformer, _expand_token
    from open_clip.model import build_vision_tower, CLIPVisionCfg
    from resnet.resnet import ResNet50
    from face import get_roi_presets, VideoTransformer
    from astfm.transformer import ASTModel
    from astfm.emo_ast import EmoAST
    from helpers import get_best_checkpoint
    

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def resize_pos_embed(state_dict, model, interpolation: str = 'bicubic', antialias: bool = True):
    # Rescale the grid of position embeddings when loading from state_dict
    old_pos_embed = state_dict.get('visual.positional_embedding', None)
    if old_pos_embed is None or not hasattr(model.visual, 'grid_size'):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = 1  # FIXME detect different token configs (ie no class token, or more)
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict['visual.positional_embedding'] = new_pos_embed


def load_state_dict(checkpoint_path: str, map_location='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, torch.jit.ScriptModule):
        state_dict = checkpoint.state_dict()
        for key in ["input_resolution", "context_length", "vocab_size"]:
            state_dict.pop(key, None)
    else:
        state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith('module'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict

def reset_layers(state_dict, keep_first):
    for key, weights in state_dict.items():
        if "resblocks" in key:
            layernum = int(key.split(".")[3])
            if layernum > keep_first :
                state_dict[key] = torch.rand_like(weights)
                print(f"Reset {key} to default weights.")
     

def load_backbone_checkpoint(model, checkpoint_path, strict=True):
    state_dict = load_state_dict(checkpoint_path)
    resize_pos_embed(state_dict, model)
    # remove text transformer related keys
    adapted_dict = {
        k: v for k, v in state_dict.items()
        if not (k.startswith("transformer.") or 
                k == "text_projection" or 
                k == "positional_embedding" or 
                k == "token_embedding.weight" or
                k == "ln_final.weight" or 
                k == "ln_final.bias"
            )
    }
    # set all weights after the given layer to be random
    incompatible_keys = model.load_state_dict(adapted_dict)
    reset_layers(adapted_dict, FREEZE_FIRST)
        
    return incompatible_keys


class TemporalTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            seq_len: int,
            embed_size: int = 512,
            heads: int = 8,
            layers: int = 12,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            output_dim: int = 512,
            no_causal_mask: bool = False,
            pool_type: str = 'tok',
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        self.num_pos = seq_len 
        self.embed_size = embed_size
        self.output_dim = output_dim
        self.heads = heads
        self.pool_type = pool_type

        self.ln_pre = norm_layer(embed_size)
        self.ln_post = norm_layer(embed_size)

        scale =  embed_size ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.zeros(embed_size))
        self.positional_embedding = nn.Parameter(scale * torch.randn(embed_size))
        self.transformer = Transformer(
            width=embed_size,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.ln_final = norm_layer(embed_size)

        if no_causal_mask:
            self.attn_mask = None
        else:
            self.register_buffer('attn_mask', self.build_causal_mask(), persistent=False)

    def build_causal_mask(self):
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.num_pos, self.num_pos)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def _global_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pool_type == 'avg':
            pooled, tokens = x[:, 1:].mean(dim=1), x[:, 1:]
        elif self.pool_type == 'tok':
            pooled, tokens = x[:, 0], x[:, 1:]
        else:
            pooled = tokens = x

        return pooled, tokens

    def forward(self, x: torch.Tensor):
        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        # shape = [*, frames+ 1, embed_size]
        x = x + self.positional_embedding.to(x.dtype)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        pooled, tokens = self._global_pool(x)
        return pooled, tokens

class CLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            quick_gelu: bool = False,
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
            freeze: bool = False
    ):
        super().__init__()
        self.freeze = freeze
        self.output_dict = output_dict

        self.visual = build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)

        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable

    def encode_image(self, image, normalize: bool = False):
        features, tokens = self.visual(image)
        features = F.normalize(features, dim=-1) if normalize else features
        return features, tokens 

    @contextmanager
    def conditional_nograd(self):
        if self.freeze:
            with torch.no_grad():
                yield 
        else:
            yield

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
    ):
        with self.conditional_nograd():
            image_features, tokens = self.encode_image(image, normalize=True) if image is not None else None

            if self.output_dict:
                out_dict = {
                    "image_features": image_features,
                    "logit_scale": self.logit_scale.exp()
                }
                if self.logit_bias is not None:
                    out_dict['logit_bias'] = self.logit_bias
                return out_dict

            if self.logit_bias is not None:
                return image_features.unsqueeze(1), tokens, self.logit_scale.exp(), self.logit_bias
            return image_features.unsqueeze(1), tokens, self.logit_scale.exp()


def get_pretrained_clip(freeze):
    net = CLIP(**MODEL_CFG, freeze=freeze)
    load_backbone_checkpoint(net, PRETRAINED_CHKPT)
    embed_dim = MODEL_CFG["embed_dim"]
    vision_width = MODEL_CFG['vision_cfg']['width']
    return net, embed_dim, vision_width


class ResNet_Base(nn.Module):
    def __init__(self, 
                 dataset_const_namespace, 
                 freeze_clip_weights=FREEZE_CLIP
    ):
        super().__init__()
        self.freeze_clip = freeze_clip_weights
        self.ds_constants = dataset_const_namespace

        self.backbone  =  ResNet50(self.ds_constants.NUM_LABELS, channels=1)

    def forward(self, a, v=None):
        out, embedding = self.backbone(a) 
        embedding_flat_norm = F.normalize(embedding, dim=1)
        return out, embedding_flat_norm


class VideoCLIPMBT(nn.Module):
    def __init__(self, 
                 dataset_const_namespace, 
                 num_bottle_token=NUM_BOT_TOKENS, 
                 num_head=NUM_ATTN_HEADS, 
                 attn_drop=ATTN_DROPOUT, 
                 num_layers=NUM_MM_LAYERS, 
                 head_base_dim=HEAD_BASE_DIM,
                 linear_drop=LINEAR_DROPOUT, 
                 freeze_clip_weights=FREEZE_CLIP
    ):
        super().__init__()
        self.freeze_clip = freeze_clip_weights
        self.ds_constants = dataset_const_namespace
        self.num_modalities = 2
        self.num_multimodal_layers = num_layers
        self.num_bottle_token = num_bottle_token
        self.vision_width = 1024

        self.backbone  = load_pretrained_videovit(PRETRAINED_CHKPT_JEPA)

        self.video_layer_sz = 1 + FRAMES + self.num_bottle_token # cls + FRAMES + tokens
        self.audio_layer_sz = 1 + 49 + self.num_bottle_token     # cls + embeddings + tokens
        self.norm_video_attn = LayerNorm(self.vision_width)
        self.norm_video_mlp = LayerNorm(self.vision_width)
        self.norm_audio_attn = LayerNorm(self.vision_width)
        self.norm_audio_mlp = LayerNorm(self.vision_width)

        layer = EncoderLayer(self.vision_width, num_head, self.vision_width, attn_drop)
        self.multimodal_audio = clones(layer, self.num_multimodal_layers)
        self.multimodal_video = clones(layer, self.num_multimodal_layers)
        self.bottleneck_token = nn.Parameter(torch.zeros(1, num_bottle_token, self.vision_width))
        self.head_base_dim = head_base_dim
        self.video_head = nn.Sequential(
            nn.Linear(self.vision_width, self.head_base_dim),
            nn.ReLU(),
            nn.Linear(self.head_base_dim, self.head_base_dim // 2),
            nn.ReLU(),
            nn.Linear(self.head_base_dim // 2, self.head_base_dim),
            nn.ReLU(),
            nn.Dropout(linear_drop),
            nn.Linear(self.head_base_dim, self.vision_width)
        ) 
        self.audio_head = nn.Sequential(
            nn.Linear(self.vision_width, self.head_base_dim),
            nn.ReLU(),
            nn.Linear(self.head_base_dim, self.head_base_dim // 2),
            nn.ReLU(),
            nn.Linear(self.head_base_dim // 2, self.head_base_dim),
            nn.ReLU(),
            nn.Dropout(linear_drop),
            nn.Linear(self.head_base_dim, self.vision_width)
        ) 

    def forward(self, a, v):
        B = a.shape[0] # Batch size
        v = self.backbone(v)
        v = F.normalize(v, dim=1)
        a = self.backbone(a)
        a = F.normalize(a, dim=1)


        bottleneck_token = self.bottleneck_token.expand(B, -1, -1)
        for i in range(self.num_multimodal_layers):
            v = torch.cat((v, bottleneck_token), dim=1)
            v = self.multimodal_video[i](self.norm_video_attn(v)) + v
            v = self.video_head(self.norm_video_mlp(v)) + v

            v_bottleneck_token = v[:, -self.num_bottle_token:]

            a = torch.cat((v_bottleneck_token, a), dim=1)
            a = self.multimodal_audio[i](self.norm_audio_attn(a)) + a
            a = self.audio_head(self.norm_audio_mlp(a)) + a

            a_bottleneck_token = a[:, :self.num_bottle_token]
            bottleneck_token = (bottleneck_token.clone() + ((a_bottleneck_token + v_bottleneck_token) / 2)) / (i + 1)

            v = v[:,  :-self.num_bottle_token]
            a = a[:,  self.num_bottle_token:]
        
        embedding = torch.cat((a[:, :1, :], v[:, :1, :]), dim=1)
        embedding_flat = embedding.flatten(start_dim=1, end_dim=2)
        embedding_flat_norm = F.normalize(embedding_flat, dim=1)
        return embedding_flat_norm

class CNN_base(nn.Module):
    def  __init__(self, kernel_width=16, stride=3, pool_sz=32, out_chans=128):
        self.audio_cnn_k_width = kernel_width
        self.audio_cnn_k_stride = stride
        self.audio_cnn_pool_sz = pool_sz
        self.audio_cnn_out_chans = out_chans

        self.audio_cnn = nn.Conv2d(
            in_channels=1, out_channels=self.audio_cnn_out_chans,
            kernel_size=(NUM_MELS, self.audio_cnn_k_width), stride=self.audio_cnn_k_stride
        )
        self.audio_cnn_relu = nn.ReLU()
        self.audio_cnn_pool = nn.MaxPool1d(kernel_size=self.audio_cnn_pool_sz)
        self.audio_cnn_flat = nn.Flatten(start_dim=1, end_dim=2)
        time_dim_sz = (int((self.ds_constants.SPEC_MAX_LEN - self.audio_cnn_k_width) / self.audio_cnn_k_stride) +1)
        time_dim_after_pool = time_dim_sz // self.audio_cnn_pool_sz
        self.audio_emb_size =  self.audio_cnn_out_chans * time_dim_after_pool

class SSAST_MSPFinetune(nn.Module):
    def __init__(self, 
                 dataset_const_namespace, 
                 pt_model_name=None,
                 linear_drop=LINEAR_DROPOUT, 
                 head_base_dim=HEAD_BASE_DIM,
    ):
        super().__init__()
        self.ast_frozen = pt_model_name is not None
        self.pretrain = not self.ast_frozen 
        self.ds_constants = dataset_const_namespace
        self.head_base_dim = head_base_dim

        self.vision_width = 768

        if pt_model_name is None:
            self.audio_backbone = ASTModel(
                    fshape=16, tshape=16, fstride=16, tstride=16,
                    input_fdim=128, input_tdim=self.ds_constants.SPEC_MAX_LEN,
                    pretrain_stage=False, load_pretrained_mdl_path=SSAST_MODEL_CHKPT)

            self.avd_head = nn.Sequential(
                nn.Linear(self.vision_width, self.head_base_dim),
                nn.ReLU(),
                nn.Linear(self.head_base_dim, 3)
            ) 
        else:
            save_path = Path(f"{SAVED_MODELS_PATH}/{pt_model_name}")
            checkpoints = list(save_path.glob("*pth"))
            best_chkpt = get_best_checkpoint(checkpoints)
            ast_state_dict = torch.load(best_chkpt, map_location=torch.device("cuda"))["ast_state_dict"]
            ast_state_dict = {
                f"module.{k}": v 
                for k, v in ast_state_dict.items()
            }
            self.audio_backbone = ASTModel(
                    fshape=16, tshape=16, fstride=16, tstride=16,
                    input_fdim=128, input_tdim=self.ds_constants.SPEC_MAX_LEN, 
                    pretrain_stage=False, pt_state_dict=ast_state_dict)
            
        
            self.audio_head = nn.Sequential(
                nn.Linear(self.vision_width, self.head_base_dim),
                nn.ReLU(),
                nn.Linear(self.head_base_dim, self.head_base_dim * 2),
                nn.ReLU(),
                nn.Linear(self.head_base_dim * 2, self.head_base_dim),
                nn.ReLU(),
                nn.Dropout(linear_drop),
                nn.Linear(self.head_base_dim, self.ds_constants.NUM_LABELS)
            ) 

    def forward(self, a):
        B = a.shape[0] # Batch size
        a_embedding, a = self.audio_backbone(a)
        a_embedding = F.normalize(a_embedding, p=2, dim=1)
        if not self.pretrain:
            out = self.audio_head(a_embedding)
            return out, a_embedding
        else:
            avd = self.avd_head(a_embedding)
            return avd, a_embedding


class SSAST_MSP(nn.Module):
    def __init__(self, 
                 dataset_const_namespace, 
                 pt_model_name=None,
                 linear_drop=LINEAR_DROPOUT, 
                 head_base_dim=HEAD_BASE_DIM,
                 emb_dim=1024
    ):
        super().__init__()
        self.ast_frozen = pt_model_name is not None
        self.pretrain = not self.ast_frozen 
        self.ds_constants = dataset_const_namespace
        self.head_base_dim = head_base_dim

        self.vision_width = 768

        if pt_model_name is None:
            self.audio_backbone = ASTModel(
                    fshape=16, tshape=16, fstride=16, tstride=16,
                    input_fdim=128, input_tdim=self.ds_constants.SPEC_MAX_LEN, emb_dim=emb_dim,
                    pretrain_stage=False, load_pretrained_mdl_path=SSAST_MODEL_CHKPT)

            self.avd_head = nn.Sequential(
                nn.Linear(self.vision_width, self.head_base_dim),
                nn.ReLU(),
                nn.Linear(self.head_base_dim, 3)
            ) 
        else:
            if pt_model_name != "base":
                save_path = Path(f"{SAVED_MODELS_PATH}/{pt_model_name}")
                checkpoints = list(save_path.glob("*pth"))
                best_chkpt = get_best_checkpoint(checkpoints)
                ast_state_dict = torch.load(best_chkpt, map_location=torch.device("cuda"))["ast_state_dict"]
                ast_state_dict = {
                    f"module.{k}": v 
                    for k, v in ast_state_dict.items()
                }
                self.audio_backbone = ASTModel(
                        fshape=16, tshape=16, fstride=16, tstride=16,
                        input_fdim=128, input_tdim=self.ds_constants.SPEC_MAX_LEN, emb_dim=emb_dim,
                        pretrain_stage=False, pt_state_dict=ast_state_dict)
            else:
                self.audio_backbone = ASTModel(
                        fshape=16, tshape=16, fstride=16, tstride=16,
                        input_fdim=128, input_tdim=self.ds_constants.SPEC_MAX_LEN, emb_dim=emb_dim,
                        pretrain_stage=False, load_pretrained_mdl_path=SSAST_MODEL_CHKPT)
        
            self.audio_head = nn.Sequential(
                nn.Linear(self.vision_width, self.head_base_dim),
                nn.ReLU(),
                nn.Linear(self.head_base_dim, self.head_base_dim * 2),
                nn.ReLU(),
                nn.Linear(self.head_base_dim * 2, self.head_base_dim),
                nn.ReLU(),
                nn.Dropout(linear_drop),
                nn.Linear(self.head_base_dim, self.ds_constants.NUM_LABELS)
            ) 

    def forward(self, a):
        B = a.shape[0] # Batch size
        a_embedding, a = self.audio_backbone(a)
        a_embedding = F.normalize(a_embedding, p=2, dim=1)
        if not self.pretrain:
            out = self.audio_head(a_embedding)
            return out, a_embedding
        else:
            avd = self.avd_head(a_embedding)
            return avd, a_embedding

    
class SSAST_Base(nn.Module):
    def __init__(self, 
                 dataset_const_namespace, 
                 pt_model_name=None,
                 linear_drop=LINEAR_DROPOUT, 
                 head_base_dim=HEAD_BASE_DIM,
    ):
        super().__init__()
        self.ast_frozen = pt_model_name is not None
        self.ds_constants = dataset_const_namespace
        self.head_base_dim = head_base_dim

        self.vision_width = 768

        if pt_model_name is None:
            self.audio_backbone = ASTModel(
                    fshape=16, tshape=16, fstride=16, tstride=16,
                    input_fdim=128, input_tdim=self.ds_constants.SPEC_MAX_LEN,
                    pretrain_stage=False, load_pretrained_mdl_path=f"{SAVED_MODELS_PATH}/SSAST-Base-Patch-400.pth")
        else:
            save_path = Path(f"{SAVED_MODELS_PATH}/{pt_model_name}")
            checkpoints = list(save_path.glob("*pth"))
            best_chkpt = get_best_checkpoint(checkpoints)
            ast_state_dict = torch.load(best_chkpt, map_location=torch.device("cuda"))["ast_state_dict"]
            ast_state_dict = {
                f"module.{k}": v 
                for k, v in ast_state_dict.items()
            }
            self.audio_backbone = ASTModel(
                    fshape=16, tshape=16, fstride=16, tstride=16,
                    input_fdim=128, input_tdim=self.ds_constants.SPEC_MAX_LEN,
                    pretrain_stage=False, pt_state_dict=ast_state_dict)
            
        
        self.audio_head = nn.Sequential(
            nn.Linear(self.vision_width, self.head_base_dim),
            nn.ReLU(),
            nn.Linear(self.head_base_dim, self.head_base_dim * 2),
            nn.ReLU(),
            nn.Linear(self.head_base_dim * 2, self.head_base_dim),
            nn.ReLU(),
            nn.Dropout(linear_drop),
            nn.Linear(self.head_base_dim, self.ds_constants.NUM_LABELS)
        ) 

    def forward(self, a):
        B = a.shape[0] # Batch size
        a_embedding, a = self.audio_backbone(a)
        a_embedding = F.normalize(a_embedding, p=2, dim=1)
        out = self.audio_head(a_embedding)
        return out, a_embedding

class SSAST_ASVP(nn.Module):
    def __init__(self, 
                 dataset_const_namespace, 
                 num_bottle_token=NUM_BOT_TOKENS, 
                 linear_drop=LINEAR_DROPOUT, 
                 num_layers=NUM_MM_LAYERS, 
                 t_out_dim=T_OUT_DIM,
                 head_base_dim=HEAD_BASE_DIM,
                 freeze_clip_weights=FREEZE_CLIP
    ):
        super().__init__()
        self.freeze_clip = freeze_clip_weights
        self.ds_constants = dataset_const_namespace
        self.classification_threshold = 0.75
        self.num_modalities = 2
        self.num_multimodal_layers = num_layers
        self.num_bottle_token = num_bottle_token
        self.out_dim = t_out_dim
        self.head_base_dim = head_base_dim

        self.vision_width = 768
        self.audio_backbone = EmoAST(
                    self.ds_constants,
                    fshape=16, tshape=16, fstride=16, tstride=16,
                    input_fdim=128, input_tdim=self.ds_constants.SPEC_MAX_LEN,
                    pretrain_stage=False, load_pretrained_mdl_path=f"{SAVED_MODELS_PATH}/utt_ssast_random.mask_pretrain.asvp/clipmbt_train_loss_0.00327.pth", freeze_first=0)
        
        self.audio_head = nn.Sequential(
            nn.Linear(self.vision_width, self.head_base_dim),
            nn.ReLU(),
            nn.Linear(self.head_base_dim, self.head_base_dim * 2),
            nn.ReLU(),
            nn.Linear(self.head_base_dim * 2, self.head_base_dim),
            nn.ReLU(),
            nn.Dropout(linear_drop),
            nn.Linear(self.head_base_dim, self.ds_constants.NUM_LABELS)
        ) 

    def forward(self, a, v):
        B = a.shape[0] # Batch size
        a_embedding, a = self.audio_backbone(a)
        out = self.audio_head(a_embedding)
        return out, a_embedding


class SSAST_Librispeech(nn.Module):
    def __init__(self, 
                 dataset_const_namespace, 
                 num_bottle_token=NUM_BOT_TOKENS, 
                 linear_drop=LINEAR_DROPOUT, 
                 num_layers=NUM_MM_LAYERS, 
                 t_out_dim=T_OUT_DIM,
                 head_base_dim=HEAD_BASE_DIM,
                 freeze_clip_weights=FREEZE_CLIP
    ):
        super().__init__()
        self.freeze_clip = freeze_clip_weights
        self.ds_constants = dataset_const_namespace
        self.classification_threshold = 0.75
        self.num_modalities = 2
        self.num_multimodal_layers = num_layers
        self.num_bottle_token = num_bottle_token
        self.out_dim = t_out_dim
        self.head_base_dim = head_base_dim
        self.audio_cnn_k_width = 16
        self.audio_cnn_k_stride = 3
        self.audio_cnn_pool_sz = 32
        self.audio_cnn_out_chans = 128

        self.audio_cnn = nn.Conv2d(
            in_channels=1, out_channels=self.audio_cnn_out_chans,
            kernel_size=(NUM_MELS, self.audio_cnn_k_width), stride=self.audio_cnn_k_stride
        )
        self.audio_cnn_relu = nn.ReLU()
        self.audio_cnn_pool = nn.MaxPool1d(kernel_size=self.audio_cnn_pool_sz)
        self.audio_cnn_flat = nn.Flatten(start_dim=1, end_dim=2)
        time_dim_sz = (int((self.ds_constants.SPEC_MAX_LEN - self.audio_cnn_k_width) / self.audio_cnn_k_stride) +1)
        time_dim_after_pool = time_dim_sz // self.audio_cnn_pool_sz
        self.audio_emb_size =  self.audio_cnn_out_chans * time_dim_after_pool

        self.vision_width = 768
        self.audio_backbone = EmoAST(
                    self.ds_constants,
                    fshape=16, tshape=16, fstride=16, tstride=16,
                    input_fdim=128, input_tdim=self.ds_constants.SPEC_MAX_LEN,
                    pretrain_stage=False, load_pretrained_mdl_path=f"{SAVED_MODELS_PATH}/utt_ast_random.mask_pretrain.librispeech/clipmbt_train_loss_0.00365.pth", freeze_first=0)
        
        self.audio_head = nn.Sequential(
            nn.Linear(self.vision_width + self.audio_emb_size, self.head_base_dim),
            # nn.Linear(self.audio_emb_size, self.head_base_dim),
            nn.ReLU(),
            # nn.Linear(self.head_base_dim, self.head_base_dim * 2),
            # nn.ReLU(),
            # nn.Linear(self.head_base_dim * 2, self.head_base_dim),
            # nn.ReLU(),
            # nn.Dropout(linear_drop),
            nn.Linear(self.head_base_dim, self.ds_constants.NUM_LABELS)
        ) 

    def forward(self, a, v):
        B = a.shape[0] # Batch size
        a_embedding, a_tokens = self.audio_backbone(a)

        a = a.permute(0, 2, 1).unsqueeze(1)
        a = self.audio_cnn(a)
        a = self.audio_cnn_relu(a)
        a = a.squeeze(2)
        a = self.audio_cnn_pool(a)
        a_cnn_embedding = self.audio_cnn_flat(a)

        combined_emb = torch.cat((a_embedding, a_cnn_embedding), dim=1)
        out = self.audio_head(combined_emb)
        return out, combined_emb

class CLIPMBT(nn.Module):
    def __init__(self, 
                 dataset_const_namespace, 
                 num_bottle_token=NUM_BOT_TOKENS, 
                 num_head=NUM_ATTN_HEADS, 
                 attn_drop=ATTN_DROPOUT, 
                 linear_drop=LINEAR_DROPOUT, 
                 num_layers=NUM_MM_LAYERS, 
                 t_out_dim=T_OUT_DIM,
                 head_base_dim=HEAD_BASE_DIM,
                 freeze_clip_weights=FREEZE_CLIP
    ):
        super().__init__()
        self.freeze_clip = freeze_clip_weights
        self.ds_constants = dataset_const_namespace
        self.classification_threshold = 0.75
        self.num_modalities = 2
        self.num_multimodal_layers = num_layers
        self.num_bottle_token = num_bottle_token
        self.out_dim = t_out_dim

        # self.backbone, self.embed_dim, self.vision_width = get_pretrained_clip(freeze=freeze_clip_weights)
        self.vision_width = 768
        self.audio_backbone = ASTModel(
                 fshape=16, tshape=16, fstride=16, tstride=16,
                 input_fdim=128, input_tdim=self.ds_constants.SPEC_MAX_LEN, model_size='base',
                 pretrain_stage=False, load_pretrained_mdl_path=f"{SAVED_MODELS_PATH}/SSAST-Base-Patch-400.pth")
        
        # self.audio_backbone = EmoAST(
        #             self.ds_constants,
        #             fshape=16, tshape=16, fstride=16, tstride=16,
        #             input_fdim=128, input_tdim=self.ds_constants.SPEC_MAX_LEN,
        #             pretrain_stage=False, load_pretrained_mdl_path=f"{SAVED_MODELS_PATH}/utt_ast_audio.only_finetune.enterface_pretrained.asvp_1/clipmbt_train_loss_1.79077.pth", freeze_first=0)



        self.video_layer_sz = 1 + FRAMES + self.num_bottle_token # cls + FRAMES + tokens
        self.audio_layer_sz = 1 + 49 + self.num_bottle_token     # cls + embeddings + tokens
        self.norm_video_attn = LayerNorm(self.vision_width)
        self.norm_video_mlp = LayerNorm(self.vision_width)
        self.norm_audio_attn = LayerNorm(self.vision_width)
        self.norm_audio_mlp = LayerNorm(self.vision_width)

        self.temporal_transformer = TemporalTransformer(self.vision_width)
        layer = EncoderLayer(self.vision_width, num_head, self.vision_width, attn_drop)
        
        self.multimodal_audio = clones(layer, self.num_multimodal_layers)
        self.multimodal_video = clones(layer, self.num_multimodal_layers)
        self.bottleneck_token = nn.Parameter(torch.zeros(1, num_bottle_token, self.vision_width))
        self.head_base_dim = head_base_dim
        self.video_head = nn.Sequential(
            nn.Linear(self.vision_width, self.head_base_dim),
            nn.ReLU(),
            nn.Linear(self.head_base_dim, self.head_base_dim // 2),
            nn.ReLU(),
            nn.Linear(self.head_base_dim // 2, self.head_base_dim),
            nn.ReLU(),
            nn.Dropout(linear_drop),
            nn.Linear(self.head_base_dim, self.vision_width)
        ) 
        self.audio_head = nn.Sequential(
            nn.Linear(self.vision_width, self.head_base_dim),
            nn.ReLU(),
            nn.Linear(self.head_base_dim, self.head_base_dim * 2),
            nn.ReLU(),
            nn.Linear(self.head_base_dim * 2, self.head_base_dim),
            # nn.Linear(self.head_base_dim // 2, self.head_base_dim),
            nn.ReLU(),
            nn.Dropout(linear_drop),
            # nn.Linear(self.head_base_dim, self.vision_width)
            nn.Linear(self.head_base_dim, self.ds_constants.NUM_LABELS)
        ) 
        self.multimodal_head = nn.Sequential(
            nn.Linear(self.num_modalities * self.vision_width, self.head_base_dim),
            nn.ReLU(),
            nn.Linear(self.head_base_dim, self.head_base_dim // 2),
            nn.ReLU(),
            nn.Linear(self.head_base_dim // 2, self.head_base_dim),
            nn.ReLU(),
            nn.Dropout(linear_drop),
            nn.Linear(self.head_base_dim, self.ds_constants.NUM_LABELS)
        ) 

    def forward(self, a, v):
        B = a.shape[0] # Batch size
        a_embedding, a = self.audio_backbone(a)
        # a = F.normalize(torch.cat((a_embedding.unsqueeze(1), a), dim=1), dim=1)

        # for i in range(self.num_multimodal_layers):
        #     a = self.multimodal_audio[i](self.norm_audio_attn(a)) + a
        #     a = self.audio_head(self.norm_audio_mlp(a)) + a

        # a_embedding = a[:, 0]
        # embedding_flat_norm = F.normalize(a_embedding, dim=1)
        out = self.audio_head(a_embedding)
        return out, a_embedding

    def _forward(self, a, v):
        frame_tokens = []
        for frame_idx in range(0, FRAMES):
            frame_batch = v[:, :, frame_idx, :, :]
            rgb_embedding, _, _  = self.backbone(frame_batch)
            frame_tokens.append(rgb_embedding)

        v = torch.cat(frame_tokens, dim=1)
        v_embedding, v = self.temporal_transformer(v)

        B = a.shape[0] # Batch size
        # a_embedding, a, _= self.backbone(a)
        a_embedding, a = self.audio_backbone(a)

        # extend last dim to 768
        a_embedding = F.pad(a_embedding, (0, self.vision_width - self.embed_dim), value=0)
        # a = F.normalize(torch.cat((a_embedding.unsqueeze(1), a), dim=1), dim=1)

        v_embedding = F.pad(v_embedding, (0, self.vision_width - self.embed_dim), value=0)
        v = F.pad(v, (0, self.vision_width - self.embed_dim), value=0)
        # v = F.normalize(torch.cat((v_embedding.unsqueeze(1), v), dim=1), dim=1)

        
        bottleneck_token = self.bottleneck_token.expand(B, -1, -1)
        for i in range(self.num_multimodal_layers):
            v = torch.cat((v, bottleneck_token), dim=1)
            v = self.multimodal_video[i](self.norm_video_attn(v)) + v
            v = self.video_head(self.norm_video_mlp(v)) + v

            v_bottleneck_token = v[:, -self.num_bottle_token:]

            a = torch.cat((v_bottleneck_token, a), dim=1)
            a = self.multimodal_audio[i](self.norm_audio_attn(a)) + a
            a = self.audio_head(self.norm_audio_mlp(a)) + a

            a_bottleneck_token = a[:, :self.num_bottle_token]
            bottleneck_token = (bottleneck_token.clone() + ((a_bottleneck_token + v_bottleneck_token) / 2)) / (i + 1)

            v = v[:,  :-self.num_bottle_token]
            a = a[:,  self.num_bottle_token:]
        
        embedding = torch.cat((a[:, :1, :], v[:, :1, :]), dim=1)
        embedding_flat = embedding.flatten(start_dim=1, end_dim=2)
        embedding_flat_norm = F.normalize(embedding_flat, dim=1)

        # embedding_avg = embedding_flat_norm.mean(dim=0)
        # embedding_flat_norm = F.normalize(embedding_flat_norm - embedding_avg, dim=1)
        # out = self.multimodal_head(embedding_flat_norm)
        # return out, embedding_flat_norm
        return embedding_flat_norm
        # embedding_flat = a_embedding.flatten(start_dim=1, end_dim=2)

        # return a_embedding

        embedding_flat_norm = F.normalize(a_embedding, dim=1)
        out = self.audio_head(embedding_flat_norm)
        return out, a_embedding

def train_supervised(network, trainldr, optimizer, scaler, loss_fn):
    total_losses = AverageMeter()
    confm = ConfusionMatrixMeter(
        num_classes=network.module.ds_constants.NUM_LABELS,
        human_readable_labels=network.module.ds_constants.LABEL_MAPPINGS
    )
    network.train()
    total_correct = 0
    total_items = 0
    for data in tqdm(trainldr):
        ast_dict = network.module.audio_backbone.state_dict()
        ast_frozen = network.module.ast_frozen 
        labels = data["labels"].flatten().to(DEVICE)
        batch_size = labels.shape[0] 

        clip0_rgb, clip0_spec = data["clip0"]

        with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=USE_AMP), torch.backends.cuda.sdp_kernel(enable_flash=False):
            output, embedding = network(clip0_spec, clip0_rgb)

            output = output.squeeze(1)
            preds = torch.argmax(output, dim=1)

            loss = loss_fn(output, labels)
            total_correct += int(torch.eq(preds, labels).sum())
            total_items += batch_size
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if ast_frozen:
            network.module.audio_backbone.load_state_dict(ast_dict)
        total_losses.update(loss.data.item(), batch_size)
        confm.update(preds.cpu(), labels.cpu())
    epoch_acc = 100 * total_correct / total_items
    return total_losses.avg(), epoch_acc, confm
    
def val_supervised(network, valldr, loss_fn):
    total_losses = AverageMeter()
    confm = ConfusionMatrixMeter(
        num_classes=network.module.ds_constants.NUM_LABELS,
        human_readable_labels=network.module.ds_constants.LABEL_MAPPINGS
    )
    network.eval()
    total_correct = 0
    total_items = 0
    for data in tqdm(valldr):
        labels = data["labels"].flatten().to(DEVICE)
        batch_size = labels.shape[0] 

        clip0_rgb, clip0_spec = data["clip0"]


        with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=USE_AMP), torch.backends.cuda.sdp_kernel(enable_flash=False), torch.no_grad():
            output, embedding = network(clip0_spec, clip0_rgb)

            output = output.squeeze(1)
            preds = torch.argmax(output, dim=1)

            loss = loss_fn(output, labels)
            total_correct += int(torch.eq(preds, labels).sum())
            total_items += batch_size
        total_losses.update(loss.data.item(), batch_size)
        confm.update(preds.cpu(), labels.cpu())
    epoch_acc = 100 * total_correct / total_items
    return total_losses.avg(), epoch_acc, confm


def train_contrastive(network, trainldr, optimizer, scaler, loss_fn):
    total_losses = AverageMeter()
    network.train()
    for data in tqdm(trainldr):
        labels = data["labels"]
        batch_size = labels.shape[0] 

        clip0_rgb, clip0_spec = data["clip0"]
        clip1_rgb, clip1_spec = data["clip1"]

        # TODO: how to do this with all frames
        with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=USE_AMP), torch.backends.cuda.sdp_kernel(enable_flash=False):
            _, clip0_features = network(clip0_spec, clip0_rgb)
            _, clip1_features = network(clip1_spec, clip1_rgb)

            if torch.isnan(clip0_features).any():
                print(clip0_spec)
                print("nan in clip0 features!")
            if torch.isnan(clip1_features).any():
                print(clip1_spec)
                print("nan in clip1 features!")


            # clip0_features = network(clip0_spec, clip0_rgb)
            # clip1_features = network(clip1_spec, clip1_rgb)

            features = torch.cat((clip0_features.unsqueeze(1), clip1_features.unsqueeze(1)), dim=1)
            loss = loss_fn(features, labels=labels, visible_labels_pct=VISIBLE_LABELS_PCT)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        total_losses.update(loss.data.item(), batch_size)
    return total_losses.avg()


def val_contrastive(network, valldr, loss_fn):
    total_losses = AverageMeter()
    network.eval()
    for data in tqdm(valldr):
        labels = data["labels"]
        batch_size = labels.shape[0] 

        clip0_rgb, clip0_spec = data["clip0"]
        clip1_rgb, clip1_spec = data["clip1"]

        # TODO: how to do this with all frames
        with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=USE_AMP), torch.backends.cuda.sdp_kernel(enable_flash=False), torch.no_grad():
            _, clip0_features = network(clip0_spec, clip0_rgb)
            _, clip1_features = network(clip1_spec, clip1_rgb)
            if torch.isnan(clip0_features).any():
                print(clip0_spec)
                print("nan in clip0 features!")
            if torch.isnan(clip1_features).any():
                print(clip1_spec)
                print("nan in clip1 features!")
            # clip0_features = network(clip0_spec, clip0_rgb)
            # clip1_features = network(clip1_spec, clip1_rgb)

            features = torch.cat((clip0_features.unsqueeze(1), clip1_features.unsqueeze(1)), dim=1)
            loss = loss_fn(features, labels=labels, visible_labels_pct=VISIBLE_LABELS_PCT)
        total_losses.update(loss.data.item(), batch_size)
    return total_losses.avg()


def train_hybrid(network, trainldr, optimizer, scaler, loss_fn):
    total_losses = AverageMeter()
    confm = ConfusionMatrixMeter(
        num_classes=network.module.ds_constants.NUM_LABELS,
        human_readable_labels=network.module.ds_constants.LABEL_MAPPINGS
    )
    network.train()
    total_correct = 0
    total_items = 0
    for data in tqdm(trainldr):
        labels = data["labels"].cuda()
        batch_size = labels.shape[0] 

        clip0_rgb, clip0_spec = data["clip0"]
        clip1_rgb, clip1_spec = data["clip1"]

        # TODO: how to do this with all frames
        with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=USE_AMP), torch.backends.cuda.sdp_kernel(enable_flash=False):
            out, clip0_features = network(clip0_spec, clip0_rgb)
            out, clip1_features = network(clip1_spec, clip1_rgb)

            out = out.squeeze(1)
            preds = torch.argmax(out, dim=1)

            total_correct += int(torch.eq(preds, labels.flatten()).sum())
            total_items += batch_size

            features = torch.cat((clip0_features.unsqueeze(1), clip1_features.unsqueeze(1)), dim=1)
            loss = loss_fn(out, features, labels, visible_labels_pct=VISIBLE_LABELS_PCT)
                    
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        total_losses.update(loss.data.item(), batch_size)
        confm.update(preds.cpu(), labels.flatten().cpu())
    epoch_acc = 100 * total_correct / total_items
    return total_losses.avg(), epoch_acc, confm


def val_hybrid(network, valldr, loss_fn):
    total_losses = AverageMeter()
    confm = ConfusionMatrixMeter(
        num_classes=network.module.ds_constants.NUM_LABELS,
        human_readable_labels=network.module.ds_constants.LABEL_MAPPINGS
    )
    network.eval()
    total_correct = 0
    total_items = 0
    for data in tqdm(valldr):
        labels = data["labels"].cuda()
        batch_size = labels.shape[0] 

        clip0_rgb, clip0_spec = data["clip0"]
        clip1_rgb, clip1_spec = data["clip1"]

        # TODO: how to do this with all frames
        with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=USE_AMP), torch.backends.cuda.sdp_kernel(enable_flash=False), torch.no_grad():
            out, clip0_features = network(clip0_spec, clip0_rgb)
            out, clip1_features = network(clip1_spec, clip1_rgb)
            out = out.squeeze(1)
            preds = torch.argmax(out, dim=1)

            total_correct += int(torch.eq(preds, labels.flatten()).sum())
            total_items += batch_size

            features = torch.cat((clip0_features.unsqueeze(1), clip1_features.unsqueeze(1)), dim=1)
            loss = loss_fn(out, features, labels, visible_labels_pct=VISIBLE_LABELS_PCT)

        total_losses.update(loss.data.item(), batch_size)
        confm.update(preds.cpu(), labels.flatten().cpu())
    epoch_acc = 100 * total_correct / total_items
    return total_losses.avg(), epoch_acc, confm


def train_contrastive_neutral_aware(network, trainldr, optimizer, scaler, loss_fn):
    total_losses = AverageMeter()
    network.train()
    for data in tqdm(trainldr):
        labels = data["labels"]
        batch_size = labels.shape[0] 

        _, spec_emo = data["clip0"]
        _, spec_neu = data["clip1"]

        # TODO: how to do this with all frames
        with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=USE_AMP), torch.backends.cuda.sdp_kernel(enable_flash=False):
            _, spec_neu_features = network(spec_neu, None)
            _, spec_emo_features = network(spec_emo, None)

            if torch.isnan(spec_neu_features).any():
                print(spec_neu)
                print("nan in spec_neu features!")
            if torch.isnan(spec_emo_features).any():
                print(spec_emo)
                print("nan in spec_emo features!")

            loss = loss_fn(spec_neu_features, spec_emo_features)
        # loss.backward()
        # optimizer.step()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        total_losses.update(loss.data.item(), batch_size)
    return total_losses.avg()


def val_contrastive_neutral_aware(network, valldr, loss_fn):
    total_losses = AverageMeter()
    network.eval()
    for data in tqdm(valldr):
        labels = data["labels"]
        batch_size = labels.shape[0] 

        _, spec_emo = data["clip0"]
        _, spec_neu = data["clip1"]

        # TODO: how to do this with all frames
        with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=USE_AMP), torch.backends.cuda.sdp_kernel(enable_flash=False), torch.no_grad():
            _, spec_neu_features = network(spec_neu, None)
            _, spec_emo_features = network(spec_emo, None)

            if torch.isnan(spec_neu_features).any():
                print(spec_neu)
                print("nan in spec_neu features!")
            if torch.isnan(spec_emo_features).any():
                print(spec_emo)
                print("nan in spec_emo features!")

            loss = loss_fn(spec_neu_features, spec_emo_features)
        total_losses.update(loss.data.item(), batch_size)
    return total_losses.avg()


def train_hybrid_neutral_aware(network, trainldr, optimizer, scaler, loss_fn):
    total_losses = AverageMeter()
    confm = ConfusionMatrixMeter(
        num_classes=network.module.ds_constants.NUM_LABELS,
        human_readable_labels=network.module.ds_constants.LABEL_MAPPINGS
    )
    network.train()
    total_correct = 0
    total_items = 0
    for data in tqdm(trainldr):
        labels = data["labels"].cuda()
        batch_size = labels.shape[0] 

        _, spec_emo = data["clip0"]
        _, spec_neu = data["clip1"]

        # TODO: how to do this with all frames
        with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=USE_AMP), torch.backends.cuda.sdp_kernel(enable_flash=False):
            out, spec_emo_features = network(spec_emo, None)
            _ , spec_neu_features = network(spec_neu, None)

            out = out.squeeze(1)
            preds = torch.argmax(out, dim=1)

            total_correct += int(torch.eq(preds, labels.flatten()).sum())
            total_items += batch_size

            loss = loss_fn(out, spec_neu_features, spec_emo_features, labels)
                    
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        total_losses.update(loss.data.item(), batch_size)
        confm.update(preds.cpu(), labels.flatten().cpu())
    epoch_acc = 100 * total_correct / total_items
    return total_losses.avg(), epoch_acc, confm


def val_hybrid_neutral_aware(network, valldr, loss_fn):
    total_losses = AverageMeter()
    confm = ConfusionMatrixMeter(
        num_classes=network.module.ds_constants.NUM_LABELS,
        human_readable_labels=network.module.ds_constants.LABEL_MAPPINGS
    )
    network.eval()
    total_correct = 0
    total_items = 0
    for data in tqdm(valldr):
        labels = data["labels"].cuda()
        batch_size = labels.shape[0] 

        _, spec_emo = data["clip0"]
        _, spec_neu = data["clip1"]

        # TODO: how to do this with all frames
        with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=USE_AMP), torch.backends.cuda.sdp_kernel(enable_flash=False), torch.no_grad():
            out, spec_emo_features = network(spec_emo, None)
            _, spec_neu_features = network(spec_neu, None)
            out = out.squeeze(1)
            preds = torch.argmax(out, dim=1)

            total_correct += int(torch.eq(preds, labels.flatten()).sum())
            total_items += batch_size

            loss = loss_fn(out, spec_neu_features, spec_emo_features, labels)

        total_losses.update(loss.data.item(), batch_size)
        confm.update(preds.cpu(), labels.flatten().cpu())
    epoch_acc = 100 * total_correct / total_items
    return total_losses.avg(), epoch_acc, confm


def pretrain_avd_guided_contrastive(network, trainldr, optimizer, scaler, loss_fn):
    total_losses = AverageMeter()
    network.train()
    for data in tqdm(trainldr):
        avd_labels = data["act_val_dom"].cuda()
        batch_size = avd_labels.shape[0] 

        audio = data["audios"]
        audio_aug = data["audios_aug"]

        # TODO: how to do this with all frames
        with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=USE_AMP), torch.backends.cuda.sdp_kernel(enable_flash=False):
            avd, audio_features = network(audio)
            avd_aug, audio_aug_features = network(audio_aug)

            if torch.isnan(audio_features).any():
                print(audio)
                print("nan in audio features!")
            if torch.isnan(audio_aug_features).any():
                print(audio_aug)
                print("nan in audio_aug features!")


            features = torch.cat((audio_features.unsqueeze(1), audio_aug_features.unsqueeze(1)), dim=1)
            avd_preds = (avd + avd_aug) / 2
            loss = loss_fn(avd_preds, features, avd_labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        total_losses.update(loss.data.item(), batch_size)
    return total_losses.avg()

def train_supervised_msp(network, trainldr, optimizer, scaler, loss_fn, freeze_pretrained_weights=True):
    total_losses = AverageMeter()
    confm = ConfusionMatrixMeter(
        num_classes=network.module.ds_constants.NUM_LABELS,
        human_readable_labels=network.module.ds_constants.LABEL_MAPPINGS
    )
    network.train()
    total_correct = 0
    total_items = 0
    for data in tqdm(trainldr):
        ast_dict = network.module.audio_backbone.state_dict()
        labels = data["labels"].flatten().to(DEVICE)
        batch_size = labels.shape[0] 

        audios = data["audios"]

        with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=USE_AMP), torch.backends.cuda.sdp_kernel(enable_flash=False):
            output, embedding = network(audios)

            output = output.squeeze(1)
            preds = torch.argmax(output, dim=1)
            loss = loss_fn(output, labels)
            total_correct += int(torch.eq(preds, labels).sum())
            total_items += batch_size
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        if freeze_pretrained_weights:
            network.module.audio_backbone.load_state_dict(ast_dict)
        total_losses.update(loss.data.item(), batch_size)
        confm.update(preds.cpu(), labels.cpu())
    epoch_acc = 100 * total_correct / total_items
    return total_losses.avg(), epoch_acc, confm
    
def val_supervised_msp(network, valldr, loss_fn):

    total_losses = AverageMeter()
    confm = ConfusionMatrixMeter(
        num_classes=network.module.ds_constants.NUM_LABELS,
        human_readable_labels=network.module.ds_constants.LABEL_MAPPINGS
    )
    network.eval()
    total_correct = 0
    total_items = 0
    for data in tqdm(valldr):
        labels = data["labels"].flatten().to(DEVICE)
        batch_size = labels.shape[0] 

        audios = data["audios"]


        with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=USE_AMP), torch.backends.cuda.sdp_kernel(enable_flash=False), torch.no_grad():
            output, embedding = network(audios)

            output = output.squeeze(1)
            preds = torch.argmax(output, dim=1)

            loss = loss_fn(output, labels)
            total_correct += int(torch.eq(preds, labels).sum())
            total_items += batch_size
        total_losses.update(loss.data.item(), batch_size)
        confm.update(preds.cpu(), labels.cpu())
    epoch_acc = 100 * total_correct / total_items
    return total_losses.avg(), epoch_acc, confm

    
def train_contrastive_msp(network, trainldr, optimizer, scaler, loss_fn):
    total_losses = AverageMeter()
    network.train()
    for data in tqdm(trainldr):

        audio = data["audios"]
        audio_aug = data["audios_aug"]
        batch_size = audio.shape[0] 

        # TODO: how to do this with all frames
        with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=USE_AMP), torch.backends.cuda.sdp_kernel(enable_flash=False):
            _ , audio_features = network(audio)
            _ , audio_aug_features = network(audio_aug)

            if torch.isnan(audio_features).any():
                print(audio)
                print("nan in audio features!")
            if torch.isnan(audio_aug_features).any():
                print(audio_aug)
                print("nan in audio_aug features!")


            features = torch.cat((audio_features.unsqueeze(1), audio_aug_features.unsqueeze(1)), dim=1)
            loss = loss_fn(features, labels=None, visible_labels_pct=0)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        total_losses.update(loss.data.item(), batch_size)
    return total_losses.avg()