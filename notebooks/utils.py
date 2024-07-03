import pickle
import json
from typing import Optional
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from ipywidgets import HBox
from IPython.display import display
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode

from extpt.visualise import reduce_dimensions
from extpt.constants import *
from extpt.data import Collate_Constrastive, load_data
from extpt.train.train_utils import get_sorted_checkpoints
from extpt.datasets.utils import get_label_colours


def clean_nans(embeddings_obj):
    clean_records = []
    clean_labels = []
    clean_other_labels = []
    records = embeddings_obj["embeddings"]
    print(f"Got {records.shape[0]} records before cleaning")
    for i, record in enumerate(records):
        if not np.isnan(record).any():
            clean_records.append(record)
            clean_labels.append(embeddings_obj["labels"][i])
            clean_other_labels.append(embeddings_obj["other_labels"][i])
    records = np.stack(clean_records, axis=0)
    print(f"Got {records.shape[0]} records after cleaning")
    return {
        "embeddings": records,
        "labels": clean_labels,
        "other_labels": clean_other_labels
    }

def points2dataframe(points, labels, other_labels, label_of_interest):
    columns = ["x", "y"]
    points_df = pd.DataFrame(points, columns=columns)
    labels = [", ".join(l) for l in labels]
    points_df["labels"] = labels
    if label_of_interest is not None:
        points_df[label_of_interest] = [o[label_of_interest] for o in other_labels]

    return points_df
    

def assign_colors_to_labels(ds_namespace, labels):
    ls = list(labels)
    unique_classes = sorted(set(ls))
    label_colors = get_label_colours(len(unique_classes))
    colors = []
    for l in ls:
        # color_idx = ds_namespace.LABEL_MAPPINGS.index(l)
        # colors.append(ds_namespace.LABEL_COLORS[color_idx])
        color_idx = unique_classes.index(l)
        colors.append(label_colors[color_idx])
    return colors, label_colors, unique_classes

def normalize(a):
    norm = np.linalg.norm(a, axis=0, keepdims=True)
    return a / norm

def get_embeddings_plot(ds_namespace, embeddings_cleaned, model_name, perplexity, apply_sif, label_of_interest, marker_sz):
    if len(embeddings_cleaned["embeddings"].shape) == 3:
        embeddings = embeddings_cleaned["embeddings"].mean(axis=1)
    else:
        embeddings = embeddings_cleaned["embeddings"]
    points2d = reduce_dimensions(embeddings, perplexity)
    if apply_sif:
        points2d = normalize(points2d)
        points2d_avg = points2d.mean(axis=0)
        points2d = points2d - points2d_avg
        
    

    embeddings_df = points2dataframe(points2d, embeddings_cleaned["labels"], embeddings_cleaned["other_labels"], label_of_interest)
    if label_of_interest is None:
        cmap, label_colors, label_mappings = assign_colors_to_labels(ds_namespace, embeddings_df["labels"])
        labels = embeddings_df["labels"]
    else:
        cmap, label_colors, label_mappings = assign_colors_to_labels(ds_namespace, embeddings_df[label_of_interest])
        labels = embeddings_df[label_of_interest]
    # hovertext_labels = pd.Series([f"{l}--{s}" for l, s in zip(labels, other_labels)])
    hovertext_labels = labels

    fig = go.Scatter(y=embeddings_df["y"], 
                     x=embeddings_df["x"],
                    #  hovertext=embeddings_df["labels"],
                     hovertext=hovertext_labels,
                     mode="markers",
                     marker=dict(
                         size=marker_sz
                     ),
                     marker_color=cmap)
    figw = go.FigureWidget([fig])
    figw.update_layout(
        autosize=False,
        width=800,
        height=800,
        template='seaborn',
        title=model_name
    )
    return figw, points2d, label_colors, label_mappings

def get_hyperparam_text(hparam_dict):
    hparam_text = ""
    for key in hparam_dict.keys():
        line = f"{key}: <b>{hparam_dict[key]}</b><br>"
        hparam_text += line
    print(hparam_text)
    return hparam_text

def get_hyperparam_display(hparam_dict):
    fig = go.Figure()
    fig.add_annotation(
        showarrow=False,
        text=get_hyperparam_text(hparam_dict)
    )
    fig.update_layout(template='simple_white')
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    figw = go.FigureWidget(fig)
    figw.update_layout(
        showlegend=False,
        autosize=False,
        width=800,
        height=800,
    )
    return figw

    
    
def tally_labels(ds_namespace, labels_list, label_mappings):
    # labels_list = [e.split("--")[0] for e in labels_list]
    counts = OrderedDict()
    for key in label_mappings:
        counts[key] = 0
    
    for label in labels_list:
        counts[label] += 1
    labels = []
    values = []
    for k in counts.keys():
        labels.append(k)
        values.append(counts[k])
        
    return labels, values

def display_interactive(ds_namespace, emb_obj, checkpoint_name, label_of_interest, should_display=True, perplexity=50, apply_sif=False, marker_sz=3):
    embeddings_cleaned = clean_nans(emb_obj)
    # plot = get_hyperparam_display(hparam_dict=hparams)
    plot, embeddings2d, label_colors, label_mappings = get_embeddings_plot(ds_namespace, embeddings_cleaned, checkpoint_name, perplexity, apply_sif, label_of_interest=label_of_interest, marker_sz=marker_sz)
    # Live updating bar chart for emotions within the selection
    barchartw = go.FigureWidget([go.Bar(
        marker={
            "color": label_colors
        }
    )])

    def selection_fn(trace, points, selector):
        labels = trace.hovertext[points.point_inds]
        loi_labels, loi_counts = tally_labels(ds_namespace, labels, label_mappings)
        barchartw.plotly_restyle({"x": [loi_labels], "y": [loi_counts]}, 0)
        bar_title = label_of_interest if label_of_interest is not None else "Emotion"
        barchartw.update_layout(title=f"{bar_title} breakdown in selection ({len(labels)} total)")


        
        
    graph = HBox((plot, barchartw))
    scatterplot = plot.data[0]
    scatterplot.on_selection(selection_fn)

        

    # plot.update_layout(
    #     updatemenus=[
    #         dict(
    #             type="buttons",
    #             direction="down",
    #             showactive=True,
    #             xanchor="right",
    #             yanchor="top",
    #             buttons=list(
    #                 [
    #                     dict(
    #                         label="Plot",
    #                         method="update",
    #                         args=[
    #                         {
    #                         },    
    #                         {
    #                             "template": 'seaborn',
    #                             "xaxis.visible": True,
    #                             "yaxis.visible": True,
    #                         }]
    #                     ),
    #                     dict(
    #                         label="Hparams",
    #                         method="update",
    #                         args=[
    #                         {
    #                         },
    #                         {
    #                             "template": 'simple_white',
    #                             "xaxis.visible": False,
    #                             "yaxis.visible": False,
    #                             "annotations": list(dict(
    #                                 showarrow=False,
    #                                 visible=True,
    #                                 align="left",
    #                                 x=0, y=0,
    #                                 text="hello world"
    #                             ))
    #                         }]
    #                     ),
    #                 ]
    #             ),
    #         )
    #     ]
    # )
    if should_display:
        display(graph)
    return embeddings2d


def load_and_display(ds_namespace, model_name=None, emb_obj=None, dl_set=None, label_of_interest=None, display=True, graph_title=None, apply_sif=False, legacy_labels=True, marker_sz=3):
    
    assert (model_name or emb_obj), "Need either model name or emb_obj to continue"
    assert not (model_name is None and emb_obj is None), "Only one of (model name, emb_obj) can be set at a time."

        
    graph_title = graph_title if graph_title else "FROM EMB_OBJ"
    if model_name:
        graph_title = model_name
        save_dir = f"/root/clip/saved_models/{model_name}"
        with open(f"{save_dir}/hparams.json", "r") as fh:
            hparams = json.load(fh)
        
        suffix = f"_{dl_set}set" if dl_set else ''
        best_chkpt =  get_sorted_checkpoints(save_dir)[0]
        pkl_file = f"{best_chkpt.with_suffix('')}{suffix}.pkl"
        with open(pkl_file, "rb") as fh:
            embeddings_obj = pickle.load(fh)
    
    if emb_obj:
        embeddings_obj = emb_obj

    if legacy_labels:
        # change the emb_obj to match new label format
        speakerids = embeddings_obj["speaker_ids"]
        del embeddings_obj["speaker_ids"]
        other_labels_obj = [{"speaker_id": i} for i in speakerids]

        embeddings_obj["other_labels"] = other_labels_obj
    embeddings2d = display_interactive(ds_namespace, embeddings_obj, graph_title, label_of_interest, display, apply_sif=apply_sif, marker_sz=marker_sz)
    return embeddings_obj, embeddings2d


def forward_wrapper(model):
    model.a_features = []
    model.v_features = []
    def forward(x, mode):
        """Calculate embeddings or logits given a batch of input image tensors.

        Arguments:
            x {torch.tensor} -- Batch of image tensors representing faces.

        Returns:
            torch.tensor -- Batch of embedding vectors or multinomial logits.
        """
        x = model.conv2d_1a(x)
        x = model.conv2d_2a(x)
        x = model.conv2d_2b(x)



        if mode == "video":
            model.v_features.append(F.normalize(x, p=2, dim=1))
        elif mode == "audio":
            model.a_features.append(F.normalize(x, p=2, dim=1))

        x = model.maxpool_3a(x)
        x = model.conv2d_3b(x)
        x = model.conv2d_4a(x)
        x = model.conv2d_4b(x)

        if mode == "video":
            model.v_features.append(F.normalize(x, p=2, dim=2))
        elif mode == "audio":
            model.a_features.append(F.normalize(x, p=2, dim=2))

        x = model.repeat_1(x)
            
        x = model.mixed_6a(x)
        x = model.repeat_2(x)
        x = model.mixed_7a(x)
        x = model.repeat_3(x)
        x = model.block8(x)

        if mode == "video":
            model.v_features.append(F.normalize(x, p=2, dim=2))
        elif mode == "audio":
            model.a_features.append(F.normalize(x, p=2, dim=2))

        x = model.avgpool_1a(x)
        x = model.dropout(x)
        x = model.last_linear(x.view(x.shape[0], -1))
        x = F.normalize(x, p=2, dim=1)
        return x

    return forward


def __forward_wrapper(attention_module):
    def attn_forward(
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = attention_module.ln_1_kv(k_x) if hasattr(attention_module, "ln_1_kv") and k_x is not None else None
        v_x = attention_module.ln_1_kv(v_x) if hasattr(attention_module, "ln_1_kv") and v_x is not None else None
        attn, attn_weights = attention_module.attention(q_x=attention_module.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask)
        cls_attn_map = attn_weights[:, 0, 1:].view(7, 7)
        attention_module.attn_maps.append(cls_attn_map)
        # attention_module.cls_attn_map = attn_weights.view(16, 16)
        x = q_x + attention_module.ls_1(attn)
        x = x + attention_module.ls_2(attention_module.mlp(attention_module.ln_2(x)))
        return x

    return attn_forward


def _forward_wrapper(attention_module):
    def attn_forward(x, mask=None):
        B, N, C = x.shape
        qkv = attention_module.qkv(x).reshape(B, N, 3, attention_module.num_heads, C // attention_module.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, D]

        if attention_module.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(q, k, v, dropout_p=attention_module.proj_drop_prob)
                attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * attention_module.scale  # [B, num_heads, D, D]
            attn = attn.softmax(dim=-1)
            attn = attention_module.attn_drop(attn)
            print(attn.shape)
            x = (attn @ v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = attention_module.proj(x)
        x = attention_module.proj_drop(x)
        return x, attn

    return attn_forward


def ast_attn_wrapper(attention_module):
    def forward(x):
        B, N, C = x.shape
        qkv = attention_module.qkv(x).reshape(B, N, 3, attention_module.num_heads, C // attention_module.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * attention_module.scale
        attn = attn.softmax(dim=-1)
        attn = attention_module.attn_drop(attn)
        cls_attn_maps = attn[:, :, 0, 2:].view(12, 8, 45)
        attention_module.attn_maps = cls_attn_maps

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = attention_module.proj(x)
        x = attention_module.proj_drop(x)
        return x
    return forward

def display_ast_attn(spec_attn, spec, title):
    spec = spec.permute(2, 1, 0).detach().cpu()
    spec_attn = F.interpolate(spec_attn.view(1, 1, 8, 45), (spec.shape[0], spec.shape[1]), mode="bilinear").squeeze(0).view(spec.shape[0], spec.shape[1], 1).detach().cpu().numpy()
    # attn = F.interpolate(attn.view(1, 1, 16, 16), (rgb.shape[0], rgb.shape[1]), mode="bilinear").squeeze(0).view(rgb.shape[0], rgb.shape[1], 1).detach().cpu().numpy()
    plt.subplot(3, 1, 1)
    plt.imshow(spec)
    plt.subplot(3, 1, 2)
    plt.imshow(spec_attn)
    plt.subplot(3, 1, 3)
    plt.imshow(spec_attn)
    plt.imshow(spec, alpha=0.4)
    plt.suptitle(title)
    plt.show()

def display_attn(attn, rgb, spec, title):
    rgb = rgb[:, :, 7, :, :].squeeze(0).permute(1, 2, 0).detach().cpu()
    spec = spec.squeeze(0).permute(1, 2, 0).detach().cpu()
    rgb_attn, spec_attn = attn[8], attn[-1]
    spec_attn = F.interpolate(spec_attn.view(1, 1, 7, 7), (rgb.shape[0], rgb.shape[1]), mode="bilinear").squeeze(0).view(rgb.shape[0], rgb.shape[1], 1).detach().cpu().numpy()
    rgb_attn = F.interpolate(rgb_attn.view(1, 1, 7, 7), (rgb.shape[0], rgb.shape[1]), mode="bilinear").squeeze(0).view(rgb.shape[0], rgb.shape[1], 1).detach().cpu().numpy()
    # attn = F.interpolate(attn.view(1, 1, 16, 16), (rgb.shape[0], rgb.shape[1]), mode="bilinear").squeeze(0).view(rgb.shape[0], rgb.shape[1], 1).detach().cpu().numpy()
    plt.subplot(3, 2, 1)
    plt.imshow(rgb)
    plt.imshow(rgb_attn, alpha=0.6)
    plt.subplot(3, 2, 2)
    plt.imshow(spec_attn)
    plt.imshow(spec, alpha=0.4)
    plt.subplot(3, 2, 3)
    plt.imshow(rgb)
    plt.subplot(3, 2, 4)
    plt.imshow(spec)
    plt.subplot(3, 2, 5)
    plt.imshow(rgb_attn)
    plt.subplot(3, 2, 6)
    plt.imshow(spec_attn)
    plt.suptitle(title)
    plt.show()


def display_cnn_feature_map(feature_maps, rgb, spec, title):
    rgb = rgb[:, :, 7, :, :].squeeze(0).permute(1, 2, 0).detach().cpu()
    spec = spec.squeeze(0).permute(1, 2, 0).detach().cpu()
    a_feature_map, v_feature_map = feature_maps
    batch, filters, height, width = a_feature_map.shape
    spec_features = a_feature_map[0, 35, :, :]
    rgb_features = v_feature_map[0, 35, :, :]
    # spec_features = a_feature_map.mean(dim=1, keepdim=True)
    # rgb_features = v_feature_map.mean(dim=1, keepdim=True)
    spec_features = F.interpolate(spec_features.view(1, 1, height, width), (rgb.shape[0], rgb.shape[1]), mode="bilinear").squeeze(0).view(rgb.shape[0], rgb.shape[1], 1).detach().cpu().numpy()
    rgb_features = F.interpolate(rgb_features.view(1, 1, height, width), (rgb.shape[0], rgb.shape[1]), mode="bilinear").squeeze(0).view(rgb.shape[0], rgb.shape[1], 1).detach().cpu().numpy()
    # features = F.interpolate(features.view(1, 1, 16, 16), (rgb.shape[0], rgb.shape[1]), mode="bilinear").squeeze(0).view(rgb.shape[0], rgb.shape[1], 1).detach().cpu().numpy()
    plt.subplot(3, 2, 1)
    plt.imshow(rgb)
    plt.imshow(rgb_features, alpha=0.6)
    plt.subplot(3, 2, 2)
    plt.imshow(spec_features)
    plt.imshow(spec, alpha=0.4)
    plt.subplot(3, 2, 3)
    plt.imshow(rgb)
    plt.subplot(3, 2, 4)
    plt.imshow(spec)
    plt.subplot(3, 2, 5)
    plt.imshow(rgb_features)
    plt.subplot(3, 2, 6)
    plt.imshow(spec_features)
    plt.suptitle(title)
    plt.show()

def display_temporal_attn(attn, temporal_input, title):
    temporal_input = temporal_input[:, 7, :].squeeze(0).view(16, 32).detach().cpu()
    temp_h, temp_w = temporal_input.shape
    attn = F.interpolate(attn.view(1, 1, 16, 16), (temp_h, temp_w), mode="bilinear").squeeze(0).view(temp_h, temp_w, 1).detach().cpu()
    plt.subplot(3, 1, 1)
    plt.imshow(temporal_input)
    plt.subplot(3, 1, 2)
    plt.imshow(attn)
    plt.subplot(3, 1, 3)
    plt.imshow(temporal_input)
    plt.imshow(attn, alpha=0.4)
    plt.suptitle(title)
    plt.show()


# def get_inference_batch(dl_set = "train"):
#     collate_fn = Collate_Constrastive(enterface)
#     train_dl, val_dl, _, _ = load_data(enterface, 
#                                         nlines=1,
#                                         batch_sz=1,
#                                         train_val_test_split=[1.0, 0.0, 0.0],
#                                         train_collate_func=collate_fn,
#                                         val_collate_func=collate_fn,
#                                         shuffle_manifest=True,
#                                         seed=None)

#     split_to_use = train_dl if dl_set=="train" else val_dl

#     data = next(iter(split_to_use))
#     clip0_rgb, clip0_spec = data["clip0"]

#     return clip0_rgb.to(DEVICE), clip0_spec.to(DEVICE)

# def load_and_display_attn(model_name, inference_batch, graph_title=None):

#     graph_title = model_name if graph_title is None else graph_title
#     save_dir = f"/root/clip/saved_models/{model_name}"
#     with open(f"{save_dir}/hparams.json", "r") as fh:
#         hparams = json.load(fh)
    
#     cmbt = CLIPMBT(enterface)
#     # cmbt = ResCLIPMBT(enterface)
#     cmbt = nn.DataParallel(cmbt).to(DEVICE)
#     scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
#     best_chkpt =  get_sorted_checkpoints(save_dir)[0].resolve()
#     save_items = torch.load(best_chkpt)
#     print(f"Loading from checkpoint {best_chkpt}")
#     model_state_dict = save_items["model_state_dict"]
#     scaler_dict = save_items["scaler_dict"]
#     cmbt.load_state_dict(model_state_dict)
#     scaler.load_state_dict(scaler_dict)
#     # print([child for child in cmbt.module.named_children()])
#     # return
#     block = -1
#     # module =  cmbt.module.backbone.visual.transformer.resblocks[block]
#     # module =  cmbt.module.backbone.blocks[block].attn
#     module =  cmbt.module.audio_backbone.v.blocks[block].attn
#     # module.attn_maps = []
#     # module.forward = forward_wrapper(module)
#     # module =  cmbt.module.backbone
#     # module.forward = forward_wrapper(module)
#     module.forward = ast_attn_wrapper(module)

#     clip0_rgb, clip0_spec = inference_batch 
#     with torch.no_grad():
#         cmbt(clip0_spec.cuda(), clip0_rgb.cuda())
#     cls_attn = module.attn_maps[8, :, :] # select attention head
#     print(cls_attn.shape)
#     # v_features = module.v_features_2b
#     # a_features = module.a_features_2b
#     # v_features = module.v_features_4b
#     # a_features = module.a_features_4b
#     # depth = 1
#     # v_features = module.v_features[depth]
#     # a_features = module.a_features[depth]
#     # print(v_features.shape)

#     # display_attn(cls_attn, clip0_rgb, clip0_spec, title=graph_title)
#     display_ast_attn(cls_attn, clip0_spec, title=graph_title)
#     # display_temporal_attn(cls_attn, tt_input, title=graph_title)
#     # display_cnn_feature_map([a_features, v_features], clip0_rgb, clip0_spec, title=graph_title)





    