# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr_vae import build as build_vae
from .detr_vae import build_cnnmlp as build_cnnmlp

def build_ACT_model(args, config):
    return build_vae(args, config)

def build_CNNMLP_model(args):
    return build_cnnmlp(args)