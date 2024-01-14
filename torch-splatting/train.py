import torch
import numpy as np
import gaussian_splatting.utils as utils
from gaussian_splatting.trainer import Trainer
import gaussian_splatting.utils.loss_utils as loss_utils
from gaussian_splatting.utils.data_utils import read_all
from gaussian_splatting.utils.camera_utils import to_viewpoint_camera
from gaussian_splatting.utils.point_utils import get_point_clouds
from gaussian_splatting.gauss_model import GaussModel
from gaussian_splatting.gauss_render import GaussRenderer

import contextlib

from torch.profiler import profile, ProfilerActivity

USE_GPU_PYTORCH = True
USE_PROFILE = False

class GSSTrainer(Trainer):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.data = kwargs.get('data') # Nirit - dataset as original :{'cam', 'im', 'seg','id'}
        self.gaussRender = GaussRenderer(**kwargs.get('render_kwargs', {}))
        self.lambda_dssim = 0.2
        self.lambda_depth = 0.0
    
    def on_train_step(self):
        ind = np.random.choice(len(self.data['id']))#np.random.choice(len(self.data['camera']))
        camera = self.data['cam'][ind]
        rgb = self.data['im'][ind]
        #depth = self.data['depth'][ind] Nirit: dont have depth
        #mask = (self.data['alpha'][ind] > 0.5) NiritL I dont have alpha
        if USE_GPU_PYTORCH:
            camera = to_viewpoint_camera(camera)

        if USE_PROFILE:
            prof = profile(activities=[ProfilerActivity.CUDA], with_stack=True)
        else:
            prof = contextlib.nullcontext()

        with prof:
            out = self.gaussRender(pc=self.model, camera=camera) # I render the fetures

        if USE_PROFILE:
            print(prof.key_averages(group_by_stack_n=True).table(sort_by='self_cuda_time_total', row_limit=20))

        # think about the loss
        fetures_im = decoder_write_it(rgb)
        l1_loss = loss_utils.l1_loss(out['render'], fetures_im)
        #depth_loss = loss_utils.l1_loss(out['depth'][..., 0][mask], depth[mask])
        #ssim_loss = 1.0-loss_utils.ssim(out['render'], rgb)

        #total_loss = (1-self.lambda_dssim) * l1_loss + self.lambda_dssim * ssim_loss + depth_loss * self.lambda_depth
        total_loss = l1_loss
        #psnr = utils.img2psnr(out['render'], rgb)
        #log_dict = {'total': total_loss,'l1':l1_loss, 'ssim': ssim_loss, 'depth': depth_loss, 'psnr': psnr}
        log_dict = {'total': total_loss,'l1':l1_loss}

        return total_loss, log_dict

    def on_evaluate_step(self, **kwargs):
        import matplotlib.pyplot as plt
        ind = np.random.choice(len(self.data['camera']))
        camera = self.data['camera'][ind]
        if USE_GPU_PYTORCH:
            camera = to_viewpoint_camera(camera)

        rgb = self.data['rgb'][ind].detach().cpu().numpy()
        out = self.gaussRender(pc=self.model, camera=camera)
        rgb_pd = out['render'].detach().cpu().numpy()
        depth_pd = out['depth'].detach().cpu().numpy()[..., 0]
        depth = self.data['depth'][ind].detach().cpu().numpy()
        depth = np.concatenate([depth, depth_pd], axis=1)
        depth = (1 - depth / depth.max())
        depth = plt.get_cmap('jet')(depth)[..., :3]
        image = np.concatenate([rgb, rgb_pd], axis=1)
        image = np.concatenate([image, depth], axis=0)
        utils.imwrite(str(self.results_folder / f'image-{self.step}.png'), image)


if __name__ == "__main__":
    '''device = 'cuda'
    #folder = '/storage/group/gaoshh/huangbb/abo_train/abo_train/B075X65R3X'
    folder = 'restarization/torch-splatting-main/B075X65R3X'
    data = read_all(folder, resize_factor=0.5)
    data = {k: v.to(device) for k, v in data.items()}
    data['depth_range'] = torch.Tensor([[1,3]]*len(data['rgb'])).to(device)


    points = get_point_clouds(data['camera'], data['depth'], data['alpha'], data['rgb'])
    raw_points = points.random_sample(2**14)
    # raw_points.write_ply(open('points.ply', 'wb'))'''# Nirit: I commented this out because  I alreadu have the point clouds

    gaussModel = GaussModel(sh_degree=4, debug=False) # Nirit: I need to change it so onlt f will be a parameter
    gaussModel.create_from_pcd(pcd=raw_points)
    
    render_kwargs = {
        'white_bkgd': True,
        'original_im:' : data['rgb'],# change this to the original image just need a sample
    }

    trainer = GSSTrainer(model=gaussModel, 
        data=data,
        train_batch_size=1, 
        train_num_steps=25000,
        i_image =100,
        train_lr=1e-3, 
        amp=False,
        fp16=False,
        results_folder='result/test',
        render_kwargs=render_kwargs,
    )

    trainer.on_evaluate_step()
    trainer.train()