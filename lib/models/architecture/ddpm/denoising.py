from copy import deepcopy

import torch
import torch.nn as nn
from mmcv.cnn.bricks.conv_module import ConvModule

from mmgen.models.architectures.ddpm.modules import TimeEmbedding, EmbedSequential
from mmgen.models.architectures.ddpm.denoising import DenoisingUnet
from mmgen.models.builder import MODULES, build_module
from mmgen.models.architectures.common import get_module_device

from torch.nn.parallel.distributed import DistributedDataParallel

from lib.ops import morton3D, morton3D_invert, packbits
from lib import get_cam_rays, module_requires_grad, custom_meshgrid
from lib.core.utils.multiplane_pos import REGULAR_POSES


@MODULES.register_module()
class DenoisingUnetMod(DenoisingUnet):

    def __init__(self,
                 image_size,
                 in_channels=3,
                 concat_cond_channels=0,
                 base_channels=128,
                 resblocks_per_downsample=3,
                 num_timesteps=1000,
                 use_rescale_timesteps=True,
                 dropout=0,
                 embedding_channels=-1,
                 num_classes=0,
                 channels_cfg=None,
                 groups=1,
                 norm_cfg=dict(type='GN', num_groups=32),
                 act_cfg=dict(type='SiLU', inplace=False),
                 shortcut_kernel_size=1,
                 use_scale_shift_norm=False,
                 num_heads=4,
                 time_embedding_mode='sin',
                 time_embedding_cfg=None,
                 resblock_cfg=dict(type='DenoisingResBlockMod'),
                 attention_cfg=dict(type='MultiHeadAttentionMod'),
                 downsample_conv=True,
                 upsample_conv=True,
                 downsample_cfg=dict(type='DenoisingDownsampleMod'),
                 upsample_cfg=dict(type='DenoisingUpsampleMod'),
                 attention_res=[16, 8],
                 pretrained=None):
        super(DenoisingUnet, self).__init__()

        self.num_classes = num_classes
        self.num_timesteps = num_timesteps
        self.use_rescale_timesteps = use_rescale_timesteps

        out_channels = in_channels
        self.out_channels = out_channels
        self.concat_cond_channels = concat_cond_channels

        # check type of image_size
        if isinstance(image_size, list) or isinstance(image_size, tuple):
            assert len(image_size) == 2, 'The length of `image_size` should be 2.'
        elif isinstance(image_size, int):
            image_size = [image_size, image_size]
        else:
            raise TypeError('Only support `int` and `list[int]` for `image_size`.')
        self.image_size = image_size

        if isinstance(channels_cfg, list):
            self.channel_factor_list = channels_cfg
        else:
            raise ValueError('Only support list or dict for `channels_cfg`, '
                             f'receive {type(channels_cfg)}')

        embedding_channels = base_channels * 4 \
            if embedding_channels == -1 else embedding_channels
        self.time_embedding = TimeEmbedding(
            base_channels,
            embedding_channels=embedding_channels,
            embedding_mode=time_embedding_mode,
            embedding_cfg=time_embedding_cfg,
            act_cfg=act_cfg)

        if self.num_classes != 0:
            self.label_embedding = nn.Embedding(self.num_classes,
                                                embedding_channels)

        self.resblock_cfg = deepcopy(resblock_cfg)
        self.resblock_cfg.setdefault('dropout', dropout)
        self.resblock_cfg.setdefault('groups', groups)
        self.resblock_cfg.setdefault('norm_cfg', norm_cfg)
        self.resblock_cfg.setdefault('act_cfg', act_cfg)
        self.resblock_cfg.setdefault('embedding_channels', embedding_channels)
        self.resblock_cfg.setdefault('use_scale_shift_norm',
                                     use_scale_shift_norm)
        self.resblock_cfg.setdefault('shortcut_kernel_size',
                                     shortcut_kernel_size)

        # get scales of ResBlock to apply attention
        attention_scale = [min(image_size) // int(res) for res in attention_res]
        self.attention_cfg = deepcopy(attention_cfg)
        self.attention_cfg.setdefault('num_heads', num_heads)
        self.attention_cfg.setdefault('groups', groups)
        self.attention_cfg.setdefault('norm_cfg', norm_cfg)

        self.downsample_cfg = deepcopy(downsample_cfg)
        self.downsample_cfg.setdefault('groups', groups)
        self.downsample_cfg.setdefault('with_conv', downsample_conv)
        self.upsample_cfg = deepcopy(upsample_cfg)
        self.upsample_cfg.setdefault('groups', groups)
        self.upsample_cfg.setdefault('with_conv', upsample_conv)

        # init the channel scale factor
        scale = 1
        self.in_blocks = nn.ModuleList([
            EmbedSequential(
                nn.Conv2d(in_channels + concat_cond_channels, base_channels, 3, 1, padding=1, groups=groups))
        ])
        self.in_channels_list = [base_channels]

        # construct the encoder part of Unet
        for level, factor in enumerate(self.channel_factor_list):
            in_channels_ = base_channels if level == 0 \
                else base_channels * self.channel_factor_list[level - 1]
            out_channels_ = base_channels * factor

            for _ in range(resblocks_per_downsample):
                layers = [
                    build_module(self.resblock_cfg, {
                        'in_channels': in_channels_,
                        'out_channels': out_channels_
                    })
                ]
                in_channels_ = out_channels_

                if scale in attention_scale:
                    layers.append(
                        build_module(self.attention_cfg,
                                     {'in_channels': in_channels_}))

                self.in_channels_list.append(in_channels_)
                self.in_blocks.append(EmbedSequential(*layers))

            if level != len(self.channel_factor_list) - 1:
                self.in_blocks.append(
                    EmbedSequential(
                        build_module(self.downsample_cfg,
                                     {'in_channels': in_channels_})))
                self.in_channels_list.append(in_channels_)
                scale *= 2

        # construct the bottom part of Unet
        self.mid_blocks = EmbedSequential(
            build_module(self.resblock_cfg, {'in_channels': in_channels_}),
            build_module(self.attention_cfg, {'in_channels': in_channels_}),
            build_module(self.resblock_cfg, {'in_channels': in_channels_}),
        )

        # construct the decoder part of Unet
        in_channels_list = deepcopy(self.in_channels_list)
        self.out_blocks = nn.ModuleList()
        for level, factor in enumerate(self.channel_factor_list[::-1]):
            for idx in range(resblocks_per_downsample + 1):
                layers = [
                    build_module(
                        self.resblock_cfg, {
                            'in_channels':
                            in_channels_ + in_channels_list.pop(),
                            'out_channels': base_channels * factor
                        })
                ]
                in_channels_ = base_channels * factor
                if scale in attention_scale:
                    layers.append(
                        build_module(self.attention_cfg,
                                     {'in_channels': in_channels_}))
                if (level != len(self.channel_factor_list) - 1
                        and idx == resblocks_per_downsample):
                    layers.append(
                        build_module(self.upsample_cfg,
                                     {'in_channels': in_channels_}))
                    scale //= 2
                self.out_blocks.append(EmbedSequential(*layers))

        self.out = ConvModule(
            in_channels=in_channels_,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            groups=groups,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            bias=True,
            order=('norm', 'act', 'conv'))

        self.init_weights(pretrained)

    def get_init_density_grid(self, num_scenes, device=None):
        grid_size = 64
        return torch.zeros(
            grid_size ** 3 if num_scenes is None else (num_scenes, 64 ** 3),
            device=device, dtype=torch.float16)

    def get_init_density_bitfield(self, num_scenes, device=None):
        grid_size = 64
        return torch.zeros(
            grid_size ** 3 // 8 if num_scenes is None else (num_scenes, 64 ** 3 // 8),
            device=device, dtype=torch.uint8)

    def update_extra_state(self, decoder, code, density_grid, density_bitfield,
                           iter_density, density_thresh=0.01, decay=0.9, S=128):
        grid_size = 64

        with torch.no_grad():
            device = get_module_device(self)
            num_scenes = density_grid.size(0)
            tmp_grid = torch.full_like(density_grid, -1)
            if isinstance(decoder, DistributedDataParallel):
                decoder = decoder.module

            # full update.
            if iter_density < 16:
                X = torch.arange(grid_size, dtype=torch.int32, device=device).split(S)
                Y = torch.arange(grid_size, dtype=torch.int32, device=device).split(S)
                Z = torch.arange(grid_size, dtype=torch.int32, device=device).split(S)

                for xs in X:
                    for ys in Y:
                        for zs in Z:
                            # construct points
                            xx, yy, zz = custom_meshgrid(xs, ys, zs)
                            coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)],
                                               dim=-1)  # [N, 3], in [0, 128)
                            indices = morton3D(coords).long()  # [N]
                            xyzs = (coords.float() - (grid_size - 1) / 2) * (2 * decoder.bound / grid_size)
                            # add noise
                            half_voxel_width = decoder.bound / grid_size
                            xyzs += torch.rand_like(xyzs) * (2 * half_voxel_width) - half_voxel_width
                            # query density
                            sigmas = decoder.point_density_decode(
                                xyzs[None].expand(num_scenes, -1, 3), code)[0].reshape(num_scenes, -1)  # (num_scenes, N)
                            # assign
                            tmp_grid[:, indices] = sigmas.clamp(
                                max=torch.finfo(tmp_grid.dtype).max).to(tmp_grid.dtype)

            # partial update (half the computation)
            else:
                N = grid_size ** 3 // 4  # H * H * H / 4
                # random sample some positions
                coords = torch.randint(0, grid_size, (N, 3), device=device)  # [N, 3], in [0, 128)
                indices = morton3D(coords).long()  # [N]
                # random sample occupied positions
                occ_indices_all = []
                for scene_id in range(num_scenes):
                    occ_indices = torch.nonzero(density_grid[scene_id] > 0).squeeze(-1)  # [Nz]
                    rand_mask = torch.randint(0, occ_indices.shape[0], [N], dtype=torch.long,
                                              device=device)
                    occ_indices_all.append(occ_indices[rand_mask])  # [Nz] --> [N], allow for duplication
                occ_indices_all = torch.stack(occ_indices_all, dim=0)
                occ_coords_all = morton3D_invert(occ_indices_all.flatten()).reshape(num_scenes, N, 3)
                indices = torch.cat([indices[None].expand(num_scenes, N), occ_indices_all], dim=0)
                coords = torch.cat([coords[None].expand(num_scenes, N, 3), occ_coords_all], dim=0)
                # same below
                xyzs = (coords.float() - (grid_size - 1) / 2) * (2 * decoder.bound / grid_size)
                half_voxel_width = decoder.bound / grid_size
                xyzs += torch.rand_like(xyzs) * (2 * half_voxel_width) - half_voxel_width
                sigmas = decoder.point_density_decode(xyzs, code)[0].reshape(num_scenes, -1)  # (num_scenes, N + N)
                # assign
                tmp_grid[torch.arange(num_scenes, device=device)[:, None], indices] = sigmas.clamp(
                    max=torch.finfo(tmp_grid.dtype).max).to(tmp_grid.dtype)

            # ema update
            valid_mask = (density_grid >= 0) & (tmp_grid >= 0)
            density_grid[:] = torch.where(valid_mask, torch.maximum(density_grid * decay, tmp_grid), density_grid)
            # density_grid[valid_mask] = torch.maximum(density_grid[valid_mask] * decay, tmp_grid[valid_mask])
            mean_density = torch.mean(density_grid.clamp(min=0))  # -1 regions are viewed as 0 density.
            iter_density += 1

            # convert to bitfield
            density_thresh = min(mean_density, density_thresh)
            packbits(density_grid, density_thresh, density_bitfield)

        return

    def get_density(self, decoder, code, cfg=dict()):
        density_thresh = cfg.get('density_thresh', 0.1)
        density_step = cfg.get('density_step', 8)
        num_scenes = code.size(0)
        device = code.device
        density_grid = self.get_init_density_grid(num_scenes, device)
        density_bitfield = self.get_init_density_bitfield(num_scenes, device)
        for i in range(density_step):
            self.update_extra_state(decoder, code, density_grid, density_bitfield, i,
                                    density_thresh=density_thresh, decay=1.0)
        return density_grid, density_bitfield

    def render(self, decoder, code, density_bitfield, h, w, intrinsics, poses, cfg=dict()):
        decoder_training_prev = decoder.training
        decoder.train(False)

        code = code.reshape(code.size(0), *(3, 6, 128, 128))

        dt_gamma_scale = cfg.get('dt_gamma_scale', 0.0)
        # (num_scenes,)
        dt_gamma = dt_gamma_scale * 2 / (intrinsics[..., 0] + intrinsics[..., 1]).mean(dim=-1)
        rays_o, rays_d = get_cam_rays(poses, intrinsics, h, w)
        num_scenes, num_imgs, h, w, _ = rays_o.size()

        rays_o = rays_o.reshape(num_scenes, num_imgs * h * w, 3)
        rays_d = rays_d.reshape(num_scenes, num_imgs * h * w, 3)
        max_render_rays = cfg.get('max_render_rays', -1)
        if 0 < max_render_rays < rays_o.size(1):
            rays_o = rays_o.split(max_render_rays, dim=1)
            rays_d = rays_d.split(max_render_rays, dim=1)
        else:
            rays_o = [rays_o]
            rays_d = [rays_d]

        out_image = []
        out_depth = []
        bg_color = 1
        for rays_o_single, rays_d_single in zip(rays_o, rays_d):
            outputs = decoder(
                rays_o_single, rays_d_single,
                code, density_bitfield, 64,
                dt_gamma=dt_gamma, perturb=False)
            weights = torch.stack(outputs['weights_sum'], dim=0) if num_scenes > 1 else outputs['weights_sum'][0]
            rgbs = (torch.stack(outputs['image'], dim=0) if num_scenes > 1 else outputs['image'][0]) \
                   + bg_color * (1 - weights.unsqueeze(-1))
            depth = torch.stack(outputs['depth'], dim=0) if num_scenes > 1 else outputs['depth'][0]
            out_image.append(rgbs)
            out_depth.append(depth)
        out_image = torch.cat(out_image, dim=1) if len(out_image) > 1 else out_image[0]
        out_depth = torch.cat(out_depth, dim=1) if len(out_depth) > 1 else out_depth[0]
        out_image = out_image.reshape(num_scenes, num_imgs, h, w, 3)
        out_depth = out_depth.reshape(num_scenes, num_imgs, h, w)

        decoder.train(decoder_training_prev)
        return out_image, out_depth

    def forward(self, x_t, t, label=None, decoder=None, concat_cond=None, return_noise=False):
        if self.use_rescale_timesteps:
            t = t.float() * (1000.0 / self.num_timesteps)
        embedding = self.time_embedding(t)

        if label is not None:
            assert hasattr(self, 'label_embedding')
            embedding = self.label_embedding(label) + embedding

        h, hs = x_t, []
        if self.concat_cond_channels > 0:
            h = torch.cat([h, concat_cond], dim=1)
        # forward downsample blocks
        for block in self.in_blocks:
            h = block(h, embedding)
            hs.append(h)

        # forward middle blocks
        h = self.mid_blocks(h, embedding)

        # forward upsample blocks
        for block in self.out_blocks:
            h = block(torch.cat([h, hs.pop()], dim=1), embedding)
        outputs = self.out(h)

        num_scenes = 8

        with module_requires_grad(decoder, False), torch.enable_grad():
            from lib.core.utils.multiplane_pos import pose_spherical
            import numpy as np

            poses = [pose_spherical(theta, phi, -1.3) for phi, theta in REGULAR_POSES]
            poses = np.stack(poses)
            pose_matrices = []

            device = 'cuda'

            fxy = torch.Tensor([131.2500, 131.2500, 64.00, 64.00])
            intrinsics = fxy.repeat(num_scenes, poses.shape[0], 1).to(device)

            for i in range(poses.shape[0]):
                M = poses[i]
                M = torch.from_numpy(M)
                M = M @ torch.Tensor([[-1, 0, 0, 0],
                                      [0, 1, 0, 0],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]]).to(M.device)

                M = torch.cat(
                    [M[:3, :3], (M[:3, 3:]) / 0.5], dim=-1)
                # M = torch.inverse(M)
                pose_matrices.append(M)

            pose_matrices = torch.stack(pose_matrices).repeat(num_scenes, 1, 1, 1).to(device)
            h, w = 128, 128

            _, den_bitfield = self.get_density(decoder, outputs.reshape(outputs.size(0), *(3, 6, 128, 128)), cfg=dict())
            print('!!!')
            print(outputs.requires_grad)
            image_multi, depth_multi = self.render(decoder, outputs, den_bitfield, h, w, intrinsics, pose_matrices,
                                                   cfg=dict())  # (num_scenes, num_imgs, h, w, 3)

            def clamp_image(img, num_images):
                images = img.permute(0, 1, 4, 2, 3).reshape(
                    num_scenes * num_images, 3, h, w)  # .clamp(min=0, max=1)
                return images
                # return torch.round(images * 255) / 255

            image_multi = clamp_image(image_multi, poses.shape[0])
            image_multi = image_multi.reshape(num_scenes, 6, 3, h, w)
            image_multi = image_multi.reshape(num_scenes, 3, 6, h, w)

        image_multi.requires_grad = True

        return image_multi
