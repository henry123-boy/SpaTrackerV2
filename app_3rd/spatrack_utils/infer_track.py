from models.SpaTrackV2.models.predictor import Predictor
import yaml
import easydict
import os
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from PIL import Image
import io
import moviepy.editor as mp
from models.SpaTrackV2.utils.visualizer import Visualizer
import tqdm
from models.SpaTrackV2.models.utils import get_points_on_a_grid
import glob
from rich import print
import argparse
import decord
from huggingface_hub import hf_hub_download

config = {
    "ckpt_dir": "Yuxihenry/SpatialTrackerCkpts",  # HuggingFace repo ID
    "cfg_dir": "config/magic_infer_offline.yaml",
}

def get_tracker_predictor(output_dir: str, vo_points: int = 756, tracker_model=None):
    """
    Initialize and return the tracker predictor and visualizer
    Args:
        output_dir: Directory to save visualization results
        vo_points: Number of points for visual odometry
    Returns:
        Tuple of (tracker_predictor, visualizer)
    """
    viz = True
    os.makedirs(output_dir, exist_ok=True)
        
    with open(config["cfg_dir"], "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = easydict.EasyDict(cfg)
    cfg.out_dir = output_dir
    cfg.model.track_num = vo_points
    
    # Check if it's a local path or HuggingFace repo
    if tracker_model is not None:
        model = tracker_model
        model.spatrack.track_num = vo_points
    else:
        if os.path.exists(config["ckpt_dir"]):
            # Local file
            model = Predictor.from_pretrained(config["ckpt_dir"], model_cfg=cfg["model"])
        else:
            # HuggingFace repo - download the model
            print(f"Downloading model from HuggingFace: {config['ckpt_dir']}")
            checkpoint_path = hf_hub_download(
                repo_id=config["ckpt_dir"],
                repo_type="model",
                filename="SpaTrack3_offline.pth"
            )
            model = Predictor.from_pretrained(checkpoint_path, model_cfg=cfg["model"])
        model.eval()
        model.to("cuda")
    
    viser = Visualizer(save_dir=cfg.out_dir, grayscale=True, 
                     fps=10, pad_value=0, tracks_leave_trace=5)

    return model, viser

def run_tracker(model, viser, temp_dir, video_name, grid_size, vo_points, fps=3):
    """
    Run tracking on a video sequence
    Args:
        model: Tracker predictor instance
        viser: Visualizer instance
        temp_dir: Directory containing temporary files
        video_name: Name of the video file (without extension)
        grid_size: Size of the tracking grid
        vo_points: Number of points for visual odometry
        fps: Frames per second for visualization
    """
    # Setup paths
    video_path = os.path.join(temp_dir, f"{video_name}.mp4")
    mask_path = os.path.join(temp_dir, f"{video_name}.png")
    out_dir = os.path.join(temp_dir, "results")
    os.makedirs(out_dir, exist_ok=True)
    
    # Load video using decord
    video_reader = decord.VideoReader(video_path)
    video_tensor = torch.from_numpy(video_reader.get_batch(range(len(video_reader))).asnumpy()).permute(0, 3, 1, 2)  # Convert to tensor and permute to (N, C, H, W)
    
    # resize make sure the shortest side is 336
    h, w = video_tensor.shape[2:]
    scale = max(336 / h, 336 / w)
    if scale < 1:
        new_h, new_w = int(h * scale), int(w * scale)
        video_tensor = T.Resize((new_h, new_w))(video_tensor)
    video_tensor = video_tensor[::fps].float()
    depth_tensor = None
    intrs = None
    extrs = None
    data_npz_load = {}
    
    # Load and process mask
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path)
        mask = cv2.resize(mask, (video_tensor.shape[3], video_tensor.shape[2]))
        mask = mask.sum(axis=-1)>0
    else:
        mask = np.ones_like(video_tensor[0,0].numpy())>0
    
    # Get frame dimensions and create grid points
    frame_H, frame_W = video_tensor.shape[2:]
    grid_pts = get_points_on_a_grid(grid_size, (frame_H, frame_W), device="cpu")
    
    # Sample mask values at grid points and filter out points where mask=0
    if os.path.exists(mask_path):
        grid_pts_int = grid_pts[0].long()
        mask_values = mask[grid_pts_int[...,1], grid_pts_int[...,0]]
        grid_pts = grid_pts[:, mask_values]
    
    query_xyt = torch.cat([torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)[0].numpy()

    # run vggt 
    if os.environ.get("VGGT_DIR", None) is not None:
        vggt_model = VGGT()
        vggt_model.load_state_dict(torch.load(VGGT_DIR))
        vggt_model.eval()
        vggt_model = vggt_model.to("cuda")
        # process the image tensor
        video_tensor = preprocess_image(video_tensor)[None]
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            # Predict attributes including cameras, depth maps, and point maps.
            aggregated_tokens_list, ps_idx = vggt_model.aggregator(video_tensor.cuda()/255)
            pose_enc = vggt_model.camera_head(aggregated_tokens_list)[-1]
            # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, video_tensor.shape[-2:])
            # Predict Depth Maps
            depth_map, depth_conf = vggt_model.depth_head(aggregated_tokens_list, video_tensor.cuda()/255, ps_idx)
            # clear the cache
            del vggt_model, aggregated_tokens_list, ps_idx, pose_enc
            torch.cuda.empty_cache()
        depth_tensor = depth_map.squeeze().cpu().numpy()
        extrs = np.eye(4)[None].repeat(len(depth_tensor), axis=0)
        extrs[:, :3, :4] = extrinsic.squeeze().cpu().numpy()
        intrs = intrinsic.squeeze().cpu().numpy()
        video_tensor = video_tensor.squeeze()
        #NOTE: 20% of the depth is not reliable
        # threshold = depth_conf.squeeze().view(-1).quantile(0.5)
        unc_metric = depth_conf.squeeze().cpu().numpy() > 0.5

    # Run model inference
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        (
            c2w_traj, intrs, point_map, conf_depth,
            track3d_pred, track2d_pred, vis_pred, conf_pred, video
        ) = model.forward(video_tensor, depth=depth_tensor,
                            intrs=intrs, extrs=extrs, 
                            queries=query_xyt,
                            fps=1, full_point=False, iters_track=4,
                            query_no_BA=True, fixed_cam=False, stage=1,
                            support_frame=len(video_tensor)-1, replace_ratio=0.2) 
        
        # Resize results to avoid too large I/O Burden
        max_size = 336
        h, w = video.shape[2:]
        scale = min(max_size / h, max_size / w)
        if scale < 1:
            new_h, new_w = int(h * scale), int(w * scale)
            video = T.Resize((new_h, new_w))(video)
            video_tensor = T.Resize((new_h, new_w))(video_tensor)
            point_map = T.Resize((new_h, new_w))(point_map)
            track2d_pred[...,:2] = track2d_pred[...,:2] * scale
            intrs[:,:2,:] = intrs[:,:2,:] * scale
            if depth_tensor is not None:
                depth_tensor = T.Resize((new_h, new_w))(depth_tensor)
            conf_depth = T.Resize((new_h, new_w))(conf_depth)
        
        # Visualize tracks
        viser.visualize(video=video[None],
                        tracks=track2d_pred[None][...,:2],
                        visibility=vis_pred[None],filename="test")
                        
        # Save in tapip3d format
        data_npz_load["coords"] = (torch.einsum("tij,tnj->tni", c2w_traj[:,:3,:3], track3d_pred[:,:,:3].cpu()) + c2w_traj[:,:3,3][:,None,:]).numpy()
        data_npz_load["extrinsics"] = torch.inverse(c2w_traj).cpu().numpy()
        data_npz_load["intrinsics"] = intrs.cpu().numpy()
        data_npz_load["depths"] = point_map[:,2,...].cpu().numpy()
        data_npz_load["video"] = (video_tensor).cpu().numpy()/255
        data_npz_load["visibs"] = vis_pred.cpu().numpy()
        data_npz_load["confs"] = conf_pred.cpu().numpy()
        data_npz_load["confs_depth"] = conf_depth.cpu().numpy()
        result_filename = f'{video_name}_result.npz'
        np.savez(os.path.join(out_dir, result_filename), **data_npz_load)

        print(f"Results saved to {out_dir}.\nTo visualize them with tapip3d, run: [bold yellow]python tapip3d_viz.py {out_dir}/{result_filename}[/bold yellow]")