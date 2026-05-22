import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import numpy as np


from ...utils.builder import TRANSFORMER
from ...utils.net_utils import xavier_init
from ...utils.transform import inverse_sigmoid
from ...utils.logger import logger
from ...utils.misc import param_size
from ...utils.points_utils import index_points
from ..bricks.point_transformers import (
    ptTransformerBlock,
    ptTransformerBlock_CrossAttn,
    SpatialTemporalTransformer
)
from ..bricks.metro_transformer import BertConfig, METROBlock
from ..bricks.pt_metro_transformer import point_METRO_block
from lib.utils.config import CN
from lib.module.ManoDecoder import ManoDecoder


@TRANSFORMER.register_module()
class PtEmbedTRv2(nn.Module):

    def __init__(self, cfg):
        super(PtEmbedTRv2, self).__init__()
        self._is_init = False

        self.nblocks = cfg.N_BLOCKS
        self.nneighbor = cfg.N_NEIGHBOR
        self.nneighbor_query = cfg.N_NEIGHBOR_QUERY
        self.nneighbor_decay = cfg.get("N_NEIGHBOR_DECAY", True)
        self.transformer_dim = cfg.TRANSFORMER_DIM
        self.feat_dim = cfg.POINTS_FEAT_DIM
        self.with_point_embed = cfg.WITH_POSI_EMBED

        self.predict_inv_sigmoid = cfg.get("PREDICT_INV_SIGMOID", False)

        self.feats_self_attn = ptTransformerBlock(self.feat_dim, self.transformer_dim, self.nneighbor)
        self.query_feats_cross_attn = nn.ModuleList()
        self.query_self_attn = nn.ModuleList()

        for i in range(self.nblocks):
            self.query_self_attn.append(ptTransformerBlock(self.feat_dim, self.transformer_dim, self.nneighbor_query))
            self.query_feats_cross_attn.append(
                ptTransformerBlock_CrossAttn(self.feat_dim,
                                             self.transformer_dim,
                                             self.nneighbor,
                                             expand_query_dim=False))

        # self.init_weights()
        logger.info(f"{type(self).__name__} has {param_size(self)}M parameters")

    def init_weights(self):
        if self._is_init == True:
            return

        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True
        logger.info(f"{type(self).__name__} init done")

    def forward(self, pt_xyz, pt_feats, query_xyz, reg_branches, query_feat=None, pt_embed=None, query_emb=None):
        if pt_embed is not None and self.with_point_embed:
            pt_feats = pt_feats + pt_embed

        if query_feat is None:
            query_feats = query_emb
        else:
            query_feats = query_feat + query_emb

        pt_feats, _ = self.feats_self_attn(pt_xyz, pt_feats)

        query_xyz_n = []
        query_feats_n = []

        # query_feats = query_emb
        for i in range(self.nblocks):
            query_feats, _ = self.query_self_attn[i](query_xyz, query_feats)

            query = torch.cat((query_xyz, query_feats), dim=-1)

            query_feats, _ = self.query_feats_cross_attn[i](pt_xyz, pt_feats, query)

            if self.predict_inv_sigmoid:
                query_xyz = reg_branches[i](query_feats) + inverse_sigmoid(query_xyz)
                query_xyz = query_xyz.sigmoid()
            else:
                query_xyz = reg_branches[i](query_feats) + query_xyz

            query_xyz_n.append(query_xyz)
            query_feats_n.append(query_feats)

        return torch.stack(query_xyz_n)
    

@TRANSFORMER.register_module()
class PtEmbedTR_Temporal(nn.Module):

    def __init__(self, cfg):
        super(PtEmbedTR_Temporal, self).__init__()
        self._is_init = False

        self.nblocks = cfg.N_BLOCKS
        self.t = cfg.WINDOW_SIZE
        self.nneighbor = cfg.N_NEIGHBOR
        self.nneighbor_query = cfg.N_NEIGHBOR_QUERY
        self.nneighbor_decay = cfg.get("N_NEIGHBOR_DECAY", True)
        self.transformer_dim = cfg.TRANSFORMER_DIM
        self.input_dim = cfg.INPUT_FEAT_DIM
        self.feat_dim = cfg.POINTS_FEAT_DIM
        self.with_point_embed = cfg.WITH_POSI_EMBED
        self.parametric_output = cfg.PARAMETRIC_OUTPUT
        self.reg_branch = nn.Sequential(nn.Linear(self.feat_dim, self.feat_dim), nn.ReLU(), nn.Linear(self.feat_dim, 3))

        self.feats_self_attn = ptTransformerBlock(self.feat_dim, self.transformer_dim, self.nneighbor)
        self.query_feats_cross_attn = nn.ModuleList()
        self.query_tempo_attn = nn.ModuleList()

        if self.parametric_output:
            # Instantiate MANO model
            self.decoder = ManoDecoder(cfg.TRANSFORMER_CENTER_IDX, 0.4, (256, 256))
            self.flat_joints = nn.Linear(21, 1)
            self.mano_linear = nn.Linear(self.input_dim, 109)  # 109 = 16 * 6 + 10 + 3, 6D theta, shape, cam

        for i in range(self.nblocks):
            self.query_tempo_attn.append(SpatialTemporalTransformer(self.feat_dim, self.transformer_dim, self.nneighbor_query, 9))
            self.query_feats_cross_attn.append(
                ptTransformerBlock_CrossAttn(self.feat_dim,
                                             self.transformer_dim,
                                             self.nneighbor,
                                             expand_query_dim=False))

        # self.init_weights()
        logger.info(f"{type(self).__name__} has {param_size(self)}M parameters")

    def init_weights(self):
        if self._is_init == True:
            return

        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True
        logger.info(f"{type(self).__name__} init done")

    def get_parametric_output(self, joints_feat, joints):
        pred_mano_params = {}
        BN = joints.size(0)
        joints_feat = joints_feat.reshape(-1, 21)
        flatten_feat = self.flat_joints(joints_feat)  # [21, B * FEAT_DIM] -> [1, B * FEAT_DIM]
        flatten_feat = flatten_feat.reshape(-1, self.input_dim)
        parametric_result = self.mano_linear(flatten_feat)  # [B, FEAT_DIM] -> [B, 106]
        pred_hand_pose = parametric_result[:, :96]
        pred_shape = parametric_result[:, 96:106]
        pred_cam = parametric_result[:, 106:]
        # positive scale
        pred_cam = torch.cat((F.relu(pred_cam[:, 0:1]), pred_cam[:, 1:]), dim=1).view(BN, 3)
        coord_xyz, coord_uv, pose_euler, shape, cam = self.decoder(pred_hand_pose, pred_shape, pred_cam)
        pred_mano_params['pose_euler'] = pose_euler
        pred_mano_params['shape'] = shape
        pred_mano_params['cam'] = cam

        # coord_uv = coord_uv / (256 // 2) - 1

        joints = coord_xyz[:, 778:]
        verts = coord_xyz[:, :778]

        return joints, verts, pred_mano_params, coord_uv

    def forward(self, pt_xyz, pt_feats, query_xyz, query_feat=None, pt_embed=None, query_emb=None):
        BT = pt_xyz.shape[0]
        T = self.t
        B = BT // T
        if pt_embed is not None and self.with_point_embed:
            pt_feats = pt_feats + pt_embed

        if query_feat is None:
            query_feats = query_emb
        else:
            query_feats = query_feat + query_emb

        pt_feats, _ = self.feats_self_attn(pt_xyz, pt_feats)

        query_xyz_n = []
        query_feats_n = []

        # query_feats = query_emb
        for i in range(self.nblocks):
            query_feats = self.query_tempo_attn[i](query_xyz, query_feats, B, T)

            query = torch.cat((query_xyz, query_feats), dim=-1)

            query_feats, _ = self.query_feats_cross_attn[i](pt_xyz, pt_feats, query)

            query_xyz = self.reg_branch(query_feat) + query_xyz

            query_xyz_n.append(query_xyz)
            query_feats_n.append(query_feats)
        
        pred_joints, pred_verts, pred_mano_params, _ = self.get_parametric_output(query_feats, query_xyz)

        # return query_xyz_n, pred_joints, pred_verts, pred_mano_params
        return torch.stack(query_xyz_n), pred_joints, pred_verts, pred_mano_params



class _Sequential(nn.Sequential):
    """
        A wrapper to allow nn.Sequential to accept multiple inputs
    """

    def forward(self, query_feats, query_xyz, pt_feats, pt_xyz):
        query_xyz_n = []
        for i, module in enumerate(self._modules.values()):
            query_feats, query_xyz, pred_verts, pred_mano_params, coord_uv = module(query_xyz, query_feats, pt_xyz, pt_feats)
            query_xyz_n.append(query_xyz)
        query_xyz_n = torch.stack(query_xyz_n)
        return query_xyz_n, pred_verts, pred_mano_params, coord_uv


@TRANSFORMER.register_module()
class PtEmbedTRv5(nn.Module):

    def __init__(self, cfg):
        super(PtEmbedTRv5, self).__init__()
        self.name = type(self).__name__
        self.cfg = cfg

        # Configuration for METRO part
        self.input_feat_dim = cfg.INPUT_FEAT_DIM
        self.hidden_feat_dim = self.input_feat_dim
        self.output_feat_dim = self.input_feat_dim
        self.dropout = cfg.DROPOUT
        self.num_hidden_layers = cfg.NUM_HIDDEN_LAYERS
        self.num_attention_heads = cfg.NUM_ATTENTION_HEADS
        # self.bps_feature_dim = cfg.BPS_FEAT_DIM
        self.parametric_output = cfg.get("PARAMETRIC_OUTPUT", False)
        self.mano_center_idx = cfg.get("TRANSFORMER_CENTER_IDX", 9)

        # Configuration for PT part
        self.nneighbor = cfg.N_NEIGHBOR
        self.nneighbor_query = cfg.N_NEIGHBOR_QUERY

        # * load metro block
        self.pt_metro_encoder = []
        self.layer_num = cfg.N_BLOCKS

        # init three transformer-encoder blocks in a loop
        for i in range(self.layer_num):
            config_class, model_class = BertConfig, point_METRO_block

            config = config_class.from_pretrained("config/backbone/bert_cfg.json")
            config.mano = cfg.MANO
            config.output_attentions = False
            config.hidden_dropout_prob = self.dropout
            config.img_feature_dim = self.input_feat_dim
            config.output_feature_dim = self.output_feat_dim
            # config.bps_feature_dim = self.bps_feature_dim + 3
            config.parametric_output = self.parametric_output
            config.center_idx = self.mano_center_idx
            self.hidden_size = self.hidden_feat_dim
            self.intermediate_size = self.hidden_size * 4

            # update model structure if specified in arguments
            update_params = ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'intermediate_size']
            for _, param in enumerate(update_params):
                arg_param = getattr(self, param)
                config_param = getattr(config, param)
                if arg_param > 0 and arg_param != config_param:
                    setattr(config, param, arg_param)

            # Required, as default value 512 < 799/4096
            config.max_position_embeddings = 778

            # Add the PT part to config
            config.n_neighbor = self.nneighbor
            config.n_neighbor_query = self.nneighbor_query
            config.init_block = False  # ! init_block won't use KNN for vec_attn
            config.final_block = True if i == self.layer_num - 1 else False  # Final block used for potential parametric output

            # init a transformer encoder and append it to a list
            assert config.hidden_size % config.num_attention_heads == 0
            model = model_class(config=config)
            self.pt_metro_encoder.append(model)

        self.pt_metro_encoder = _Sequential(*self.pt_metro_encoder)

        logger.info(f"{type(self).__name__} has {param_size(self)}M parameters")

    def forward(self, query_xyz, query_feat, pt_xyz, pt_feats):
        pred_joints, pred_verts, pred_mano_params, coord_uv = self.pt_metro_encoder(query_feats=query_feat,
                                                                         query_xyz=query_xyz,
                                                                         pt_feats=pt_feats,
                                                                         pt_xyz=pt_xyz)
        
        return pred_joints, pred_verts, pred_mano_params, coord_uv