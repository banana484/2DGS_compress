# Copyright ...
# (원본 헤더 생략)

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


# ======================================================================================
# ✔ [변경] OptimizationParams 에 pcgs_interval 추가
# ======================================================================================
class OptimizationParams(OptimizationParams):
    def __init__(self, parser):
        super().__init__(parser)
    
        self.pcgs_interval = 1


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()
        gaussians.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick random camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # ============================================================
        # ✔ 렌더링 수행
        # ============================================================
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal = render_pkg["rend_normal"]
        surf_normal = render_pkg["surf_normal"]
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        total_loss = loss + dist_loss + normal_loss
        total_loss.backward()
        iter_end.record()

  
        with torch.no_grad():

            # progress bar update
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.5f}",
                    "distort": f"{ema_dist_for_log:.5f}",
                    "normal": f"{ema_normal_for_log:.5f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                })
                progress_bar.update(10)

            # tensorboard
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss,
                            iter_start.elapsed_time(iter_end),
                            testing_iterations, scene, render, (pipe, background))

            # save model
            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            # ==========================================================
            # Densification 단계
            # ==========================================================
            if iteration < opt.densify_until_iter:

                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        opt.opacity_cull,
                        scene.cameras_extent,
                        size_threshold
                    )

                # ===================================================================
                # ✔ [추가] Per-Ray 기반 Gaussian 압축 트리거
                # ===================================================================
                if (iteration > 0 and opt.pcgs_interval > 0 and
                        iteration % opt.pcgs_interval == 0):
                    gaussians.compress_gaussians()
                    print(f"\n[ITER {iteration}] Compressed Gaussians.")
                # ===================================================================

                if (iteration % opt.opacity_reset_interval == 0 or
                    (dataset.white_background and iteration == opt.densify_from_iter)):
                    gaussians.reset_opacity()

            # optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            # checkpoint
            if iteration in checkpoint_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration),
                           scene.model_path + f"/chkpnt{iteration}.pth")

        # GUI server
        with torch.no_grad():
            if network_gui.conn is None:
                network_gui.try_connect(dataset.render_items)

            while network_gui.conn is not None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam is not None:
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)
                        net_image = render_net_image(
                            render_pkg, dataset.render_items, render_mode, custom_cam
                        )
                        net_image_bytes = memoryview(
                            (torch.clamp(net_image, 0, 1) * 255)
                            .byte().permute(1, 2, 0).contiguous().cpu().numpy()
                        )

                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                    }
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break

                except Exception:
                    network_gui.conn = None


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print("Output folder:", args.model_path)
    os.makedirs(args.model_path, exist_ok=True)

    with open(os.path.join(args.model_path, "cfg_args"), 'w') as f:
        f.write(str(Namespace(**vars(args))))

    tb_writer = SummaryWriter(args.model_path) if TENSORBOARD_FOUND else None
    return tb_writer


@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed,
                    testing_iterations, scene: Scene, renderFunc, renderArgs):

    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'test', 'cameras': scene.getTestCameras()},
            {'name': 'train',
             'cameras': [scene.getTrainCameras()[i %
                                                len(scene.getTrainCameras())]
                         for i in range(5, 30, 5)]}
        )

        for config in validation_configs:
            if not config['cameras']:
                continue

            l1_test = 0.0
            psnr_test = 0.0

            for idx, view in enumerate(config['cameras']):
                render_pkg = renderFunc(view, scene.gaussians, *renderArgs)
                image = torch.clamp(render_pkg["render"], 0, 1).to("cuda")
                gt_image = torch.clamp(view.original_image.to("cuda"), 0, 1)

                if tb_writer and idx < 5:
                    from utils.general_utils import colormap

                    depth = render_pkg["surf_depth"]
                    depth = colormap((depth / depth.max()).cpu().numpy()[0], cmap='turbo')
                    tb_writer.add_images(f"{config['name']}_view_{view.image_name}/depth",
                                         depth[None], iteration)

                    tb_writer.add_images(f"{config['name']}_view_{view.image_name}/render",
                                         image[None], iteration)

                l1_test += l1_loss(image, gt_image).mean().double()
                psnr_test += psnr(image, gt_image).mean().double()

            l1_test /= len(config['cameras'])
            psnr_test /= len(config['cameras'])

            print(f"\n[ITER {iteration}] Evaluating {config['name']}  L1:{l1_test}  PSNR:{psnr_test}")
            if tb_writer:
                tb_writer.add_scalar(f"{config['name']}/l1_loss", l1_test, iteration)
                tb_writer.add_scalar(f"{config['name']}/psnr", psnr_test, iteration)

        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--test_iterations', nargs="+", type=int,
                        default=[7000, 30000])
    parser.add_argument('--save_iterations', nargs="+", type=int,
                        default=[7000, 30000])
    parser.add_argument('--quiet', action="store_true")
    parser.add_argument('--checkpoint_iterations', nargs="+", type=int,
                        default=[])
    parser.add_argument('--start_checkpoint', type=str, default=None)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing", args.model_path)

    safe_state(args.quiet)
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint
    )

    print("\nTraining complete.")
