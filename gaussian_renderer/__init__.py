#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import math
import time
import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from upscaling.upscaler import Upscaler
import upscaling.upscaler_model


def render(
        viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
        override_color=None, use_trained_exp=False, upscaler: Upscaler = None, round_sizes=1,
        measure_time=False, use_events=True):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    full_width = int(viewpoint_camera.image_width)
    full_height = int(viewpoint_camera.image_height)
    if round_sizes != 1:
        full_width = (full_width // round_sizes) * round_sizes
        full_height = (full_height // round_sizes) * round_sizes

    subsampling_factor = 1
    if upscaler is not None:
        subsampling_factor = upscaler.get_ss_factor()
    if upscaler is not None:
        render_width, render_height = upscaler.query_render_resolution(
            full_width, full_height)
        subsampling_factor = int(round(full_width / render_width))
    else:
        render_width = full_width // subsampling_factor
        render_height = full_height // subsampling_factor

    raster_settings = GaussianRasterizationSettings(
        image_height=render_height,
        image_width=render_width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    elapsed_time_render = 0.0
    if measure_time:
        if use_events:
            start_render = torch.cuda.Event(enable_timing=True)
            end_render = torch.cuda.Event(enable_timing=True)
            start_render.record()
        else:
            torch.cuda.synchronize()
            start_time_render = time.time()
    rendered_image, radii, gradient_image, depth_image = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    if measure_time:
        if use_events:
            end_render.record()
            torch.cuda.synchronize()
            elapsed_time_render = start_render.elapsed_time(end_render) * 1e-3
        else:
            torch.cuda.synchronize()
            end_time_render = time.time()
            elapsed_time_render = end_time_render - start_time_render

    # Apply exposure to rendered image (training only)
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]

    elapsed_time_upscale = 0.0
    rendered_image_orig = None
    if upscaler is not None:
        rendered_image_orig = rendered_image
        if measure_time:
            if use_events:
                start_upscale = torch.cuda.Event(enable_timing=True)
                end_upscale = torch.cuda.Event(enable_timing=True)
                start_upscale.record()
            else:
                torch.cuda.synchronize()
                start_time_upscale = time.time()
        rendered_image = upscaler.apply(
            render_width, render_height, full_width, full_height,
            rendered_image=rendered_image_orig, depth_image=depth_image, gradient_image=gradient_image)
        if measure_time:
            if use_events:
                end_upscale.record()
                torch.cuda.synchronize()
                elapsed_time_upscale = start_upscale.elapsed_time(end_upscale) * 1e-3
            else:
                torch.cuda.synchronize()
                end_time_upscale = time.time()
                elapsed_time_upscale = end_time_upscale - start_time_upscale
        rendered_image_orig = rendered_image_orig.clamp(0, 1)
    if upscaler is None or not isinstance(upscaler, upscaling.upscaler_model.UpscalerModel):
        rendered_image = rendered_image.clamp(0, 1)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    radii = radii * subsampling_factor
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": (radii > 0).nonzero(),
        "radii": radii,
        "depth": depth_image,
        "gradient": gradient_image,
        "time_render": elapsed_time_render,
        "time_upscale": elapsed_time_upscale,
    }
    if rendered_image_orig is not None:
        rendered_image_orig = rendered_image_orig.clamp(0, 1)
        out["render_small"] = rendered_image_orig

    return out
