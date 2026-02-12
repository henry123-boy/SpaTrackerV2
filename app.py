# Disable Triton in xformers to avoid compatibility issues
import os
os.environ['XFORMERS_DISABLE_TRITON'] = '1'

import gradio as gr
import json
import numpy as np
import cv2
import base64
import time
import tempfile
import shutil
import glob
import threading
import subprocess
import struct
import zlib
from pathlib import Path
from einops import rearrange
from typing import List, Tuple, Union
try:
    import spaces   
except ImportError:
    # Fallback for local development
    def spaces(func):
        return func
import torch
import logging
from concurrent.futures import ThreadPoolExecutor
import atexit
import uuid
from models.SpaTrackV2.models.vggt4track.models.vggt_moe import VGGT4Track
from models.SpaTrackV2.models.vggt4track.utils.load_fn import preprocess_image
from models.SpaTrackV2.models.predictor import Predictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import custom modules with error handling
try:
    from app_3rd.sam_utils.inference import SamPredictor, get_sam_predictor, run_inference
    from app_3rd.spatrack_utils.infer_track import get_tracker_predictor, run_tracker, get_points_on_a_grid
except ImportError as e:
    logger.error(f"Failed to import custom modules: {e}")
    raise

# Constants
MAX_FRAMES_OFFLINE = 80
MAX_FRAMES_ONLINE = 300

COLORS = [(0, 0, 255), (0, 255, 255)]  # BGR: Red for negative, Yellow for positive
MARKERS = [1, 5]  # Cross for negative, Star for positive
MARKER_SIZE = 8

# Thread pool for delayed deletion
thread_pool_executor = ThreadPoolExecutor(max_workers=2)

def delete_later(path: Union[str, os.PathLike], delay: int = 600):
    """Delete file or directory after specified delay (default 10 minutes)"""
    def _delete():
        try:
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except Exception as e:
            logger.warning(f"Failed to delete {path}: {e}")
    
    def _wait_and_delete():
        time.sleep(delay)
        _delete()
    
    thread_pool_executor.submit(_wait_and_delete)
    atexit.register(_delete)

def create_user_temp_dir():
    """Create a unique temporary directory for each user session"""
    session_id = str(uuid.uuid4())[:8]  # Short unique ID
    temp_dir = os.path.join("temp_local", f"session_{session_id}")
    os.makedirs(temp_dir, exist_ok=True)
    
    # Schedule deletion after 10 minutes
    delete_later(temp_dir, delay=600)
    
    return temp_dir

from huggingface_hub import hf_hub_download

vggt4track_model = VGGT4Track.from_pretrained("Yuxihenry/SpatialTrackerV2_Front")
vggt4track_model.eval()
vggt4track_model = vggt4track_model.to("cuda")

# Global model initialization
print("🚀 Initializing local models...")
tracker_model_offline = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Offline")
tracker_model_offline.eval()
tracker_model_online = Predictor.from_pretrained("Yuxihenry/SpatialTrackerV2-Online")
tracker_model_online.eval() 
predictor = get_sam_predictor()
print("✅ Models loaded successfully!")

gr.set_static_paths(paths=[Path.cwd().absolute()/"_viz"]) 

@spaces.GPU
def gpu_run_inference(predictor_arg, image, points, boxes):
    """GPU-accelerated SAM inference"""
    if predictor_arg is None:
        print("Initializing SAM predictor inside GPU function...")
        predictor_arg = get_sam_predictor(predictor=predictor)
    
    # Ensure predictor is on GPU
    try:
        if hasattr(predictor_arg, 'model'):
            predictor_arg.model = predictor_arg.model.cuda()
        elif hasattr(predictor_arg, 'sam'):
            predictor_arg.sam = predictor_arg.sam.cuda()
        elif hasattr(predictor_arg, 'to'):
            predictor_arg = predictor_arg.to('cuda')
        
        if hasattr(image, 'cuda'):
            image = image.cuda()
            
    except Exception as e:
        print(f"Warning: Could not move predictor to GPU: {e}")
    
    return run_inference(predictor_arg, image, points, boxes)

@spaces.GPU
def gpu_run_tracker(tracker_model_arg, tracker_viser_arg, temp_dir, video_name, grid_size, vo_points, fps, mode="offline"):
    """GPU-accelerated tracking"""
    import torchvision.transforms as T
    import decord
    
    if tracker_model_arg is None or tracker_viser_arg is None:
        print("Initializing tracker models inside GPU function...")
        out_dir = os.path.join(temp_dir, "results")
        os.makedirs(out_dir, exist_ok=True) 
        if mode == "offline":
            tracker_model_arg, tracker_viser_arg = get_tracker_predictor(out_dir, vo_points=vo_points,
                                                                         tracker_model=tracker_model_offline.cuda())
        else:
            tracker_model_arg, tracker_viser_arg = get_tracker_predictor(out_dir, vo_points=vo_points,
                                                                         tracker_model=tracker_model_online.cuda())
    
    # Setup paths
    video_path = os.path.join(temp_dir, f"{video_name}.mp4")
    mask_path = os.path.join(temp_dir, f"{video_name}.png")
    out_dir = os.path.join(temp_dir, "results")
    os.makedirs(out_dir, exist_ok=True)
    
    # Load video using decord
    video_reader = decord.VideoReader(video_path)
    video_tensor = torch.from_numpy(video_reader.get_batch(range(len(video_reader))).asnumpy()).permute(0, 3, 1, 2)
    
    # Resize to ensure minimum side is 336
    h, w = video_tensor.shape[2:]
    scale = max(224 / h, 224 / w)
    if scale < 1:
        new_h, new_w = int(h * scale), int(w * scale)
        video_tensor = T.Resize((new_h, new_w))(video_tensor)
    if mode == "offline":
        video_tensor = video_tensor[::fps].float()[:MAX_FRAMES_OFFLINE]
    else:
        video_tensor = video_tensor[::fps].float()[:MAX_FRAMES_ONLINE]
    
    # Move to GPU
    video_tensor = video_tensor.cuda()
    print(f"Video tensor shape: {video_tensor.shape}, device: {video_tensor.device}")
    
    depth_tensor = None
    intrs = None
    extrs = None
    data_npz_load = {}

    # run vggt 
    # process the image tensor
    video_tensor = preprocess_image(video_tensor)[None]
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            # Predict attributes including cameras, depth maps, and point maps.
            predictions = vggt4track_model(video_tensor.cuda()/255)
            extrinsic, intrinsic = predictions["poses_pred"], predictions["intrs"]
            depth_map, depth_conf = predictions["points_map"][..., 2], predictions["unc_metric"]

    depth_tensor = depth_map.squeeze().cpu().numpy()
    extrs = np.eye(4)[None].repeat(len(depth_tensor), axis=0)
    extrs = extrinsic.squeeze().cpu().numpy()
    intrs = intrinsic.squeeze().cpu().numpy()
    video_tensor = video_tensor.squeeze()
    #NOTE: 20% of the depth is not reliable
    # threshold = depth_conf.squeeze()[0].view(-1).quantile(0.6).item()
    unc_metric = depth_conf.squeeze().cpu().numpy() > 0.5
    # Load and process mask
    if os.path.exists(mask_path):
        mask = cv2.imread(mask_path)
        mask = cv2.resize(mask, (video_tensor.shape[3], video_tensor.shape[2]))
        mask = mask.sum(axis=-1)>0
    else:
        mask = np.ones_like(video_tensor[0,0].cpu().numpy())>0
        grid_size = 10

    # Get frame dimensions and create grid points
    frame_H, frame_W = video_tensor.shape[2:]
    grid_pts = get_points_on_a_grid(grid_size, (frame_H, frame_W), device="cuda")
    
    # Sample mask values at grid points and filter
    if os.path.exists(mask_path):
        grid_pts_int = grid_pts[0].long()
        mask_values = mask[grid_pts_int.cpu()[...,1], grid_pts_int.cpu()[...,0]]
        grid_pts = grid_pts[:, mask_values]
    
    query_xyt = torch.cat([torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2)[0].cpu().numpy()
    print(f"Query points shape: {query_xyt.shape}")
    # Run model inference
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        (
            c2w_traj, intrs, point_map, conf_depth,
            track3d_pred, track2d_pred, vis_pred, conf_pred, video
        ) = tracker_model_arg.forward(video_tensor, depth=depth_tensor,
                            intrs=intrs, extrs=extrs, 
                            queries=query_xyt,
                            fps=1, full_point=False, iters_track=4,
                            query_no_BA=True, fixed_cam=False, stage=1, unc_metric=unc_metric,
                            support_frame=len(video_tensor)-1, replace_ratio=0.2)

        # Resize results to avoid large I/O
        max_size = 224
        h, w = video.shape[2:]
        scale = min(max_size / h, max_size / w)
        if scale < 1:
            new_h, new_w = int(h * scale), int(w * scale)
            video = T.Resize((new_h, new_w))(video)
            video_tensor = T.Resize((new_h, new_w))(video_tensor)
            point_map = T.Resize((new_h, new_w))(point_map)
            track2d_pred[...,:2] = track2d_pred[...,:2] * scale
            intrs[:,:2,:] = intrs[:,:2,:] * scale
            conf_depth = T.Resize((new_h, new_w))(conf_depth)
        
        # Visualize tracks
        tracker_viser_arg.visualize(video=video[None],
                        tracks=track2d_pred[None][...,:2],
                        visibility=vis_pred[None],filename=video_name)
                        
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

    return result_filename

def compress_and_write(filename, header, blob):
    header_bytes = json.dumps(header).encode("utf-8")
    header_len = struct.pack("<I", len(header_bytes))
    with open(filename, "wb") as f:
        f.write(header_len)
        f.write(header_bytes)
        f.write(blob)

def process_point_cloud_data(npz_file, width=256, height=192, fps=4):
    fixed_size = (width, height)
    
    data = np.load(npz_file)
    extrinsics = data["extrinsics"]
    intrinsics = data["intrinsics"]
    trajs = data["coords"]
    T, C, H, W = data["video"].shape
    
    fx = intrinsics[0, 0, 0]
    fy = intrinsics[0, 1, 1]
    fov_y = 2 * np.arctan(H / (2 * fy)) * (180 / np.pi)
    fov_x = 2 * np.arctan(W / (2 * fx)) * (180 / np.pi)
    original_aspect_ratio = (W / fx) / (H / fy)
    
    rgb_video = (rearrange(data["video"], "T C H W -> T H W C") * 255).astype(np.uint8)
    rgb_video = np.stack([cv2.resize(frame, fixed_size, interpolation=cv2.INTER_AREA)
                          for frame in rgb_video])
    
    depth_video = data["depths"].astype(np.float32)
    if "confs_depth" in data.keys():
        confs = (data["confs_depth"].astype(np.float32) > 0.5).astype(np.float32)
        depth_video = depth_video * confs
    depth_video = np.stack([cv2.resize(frame, fixed_size, interpolation=cv2.INTER_NEAREST)
                            for frame in depth_video])
    
    scale_x = fixed_size[0] / W
    scale_y = fixed_size[1] / H
    intrinsics = intrinsics.copy()
    intrinsics[:, 0, :] *= scale_x
    intrinsics[:, 1, :] *= scale_y
    
    min_depth = float(depth_video.min()) * 0.8
    max_depth = float(depth_video.max()) * 1.5
    
    depth_normalized = (depth_video - min_depth) / (max_depth - min_depth)
    depth_int = (depth_normalized * ((1 << 16) - 1)).astype(np.uint16)
    
    depths_rgb = np.zeros((T, fixed_size[1], fixed_size[0], 3), dtype=np.uint8)
    depths_rgb[:, :, :, 0] = (depth_int & 0xFF).astype(np.uint8)
    depths_rgb[:, :, :, 1] = ((depth_int >> 8) & 0xFF).astype(np.uint8)
    
    first_frame_inv = np.linalg.inv(extrinsics[0])
    normalized_extrinsics = np.array([first_frame_inv @ ext for ext in extrinsics])
    
    normalized_trajs = np.zeros_like(trajs)
    for t in range(T):
        homogeneous_trajs = np.concatenate([trajs[t], np.ones((trajs.shape[1], 1))], axis=1)
        transformed_trajs = (first_frame_inv @ homogeneous_trajs.T).T
        normalized_trajs[t] = transformed_trajs[:, :3]
    
    arrays = {
        "rgb_video": rgb_video,
        "depths_rgb": depths_rgb,
        "intrinsics": intrinsics,
        "extrinsics": normalized_extrinsics,
        "inv_extrinsics": np.linalg.inv(normalized_extrinsics),
        "trajectories": normalized_trajs.astype(np.float32),
        "cameraZ": 0.0
    }
    
    header = {}
    blob_parts = []
    offset = 0
    for key, arr in arrays.items():
        arr = np.ascontiguousarray(arr)
        arr_bytes = arr.tobytes()
        header[key] = {
            "dtype": str(arr.dtype),
            "shape": arr.shape,
            "offset": offset,
            "length": len(arr_bytes)
        }
        blob_parts.append(arr_bytes)
        offset += len(arr_bytes)
    
    raw_blob = b"".join(blob_parts)
    compressed_blob = zlib.compress(raw_blob, level=9)
    
    header["meta"] = {
        "depthRange": [min_depth, max_depth],
        "totalFrames": int(T),
        "resolution": fixed_size,
        "baseFrameRate": fps,
        "numTrajectoryPoints": normalized_trajs.shape[1],
        "fov": float(fov_y),
        "fov_x": float(fov_x),
        "original_aspect_ratio": float(original_aspect_ratio),
        "fixed_aspect_ratio": float(fixed_size[0]/fixed_size[1])
    }
    
    compress_and_write('./_viz/data.bin', header, compressed_blob)
    with open('./_viz/data.bin', "rb") as f:
        encoded_blob = base64.b64encode(f.read()).decode("ascii")
    os.unlink('./_viz/data.bin')
    
    random_path = f'./_viz/_{time.time()}.html'
    with open('./_viz/viz_template.html') as f:
        html_template = f.read()
    html_out = html_template.replace(
        "<head>",
        f"<head>\n<script>window.embeddedBase64 = `{encoded_blob}`;</script>"
    )
    with open(random_path,'w') as f:
        f.write(html_out)
    
    return random_path 

def numpy_to_base64(arr):
    """Convert numpy array to base64 string"""
    return base64.b64encode(arr.tobytes()).decode('utf-8')

def base64_to_numpy(b64_str, shape, dtype):
    """Convert base64 string back to numpy array"""
    return np.frombuffer(base64.b64decode(b64_str), dtype=dtype).reshape(shape)

def get_video_name(video_path):
    """Extract video name without extension"""
    return os.path.splitext(os.path.basename(video_path))[0]

def extract_first_frame(video_path):
    """Extract first frame from video file"""
    try:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame_rgb
        else:
            return None
    except Exception as e:
        print(f"Error extracting first frame: {e}")
        return None

def handle_video_upload(video):
    """Handle video upload and extract first frame"""
    if video is None:
        return (None, None, [], 
                gr.update(value=50), 
                gr.update(value=756), 
                gr.update(value=3))
    
    # Create user-specific temporary directory
    user_temp_dir = create_user_temp_dir()
    
    # Get original video name and copy to temp directory
    if isinstance(video, str):
        video_name = get_video_name(video)
        video_path = os.path.join(user_temp_dir, f"{video_name}.mp4")
        shutil.copy(video, video_path)
    else:
        video_name = get_video_name(video.name)
        video_path = os.path.join(user_temp_dir, f"{video_name}.mp4")
        with open(video_path, 'wb') as f:
            f.write(video.read())

    print(f"📁 Video saved to: {video_path}")
    
    # Extract first frame
    frame = extract_first_frame(video_path)
    if frame is None:
        return (None, None, [], 
                gr.update(value=50), 
                gr.update(value=756), 
                gr.update(value=3))
    
    # Resize frame to have minimum side length of 336
    h, w = frame.shape[:2]
    scale = 336 / min(h, w)
    new_h, new_w = int(h * scale)//2*2, int(w * scale)//2*2
    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Store frame data with temp directory info
    frame_data = {
        'data': numpy_to_base64(frame),
        'shape': frame.shape,
        'dtype': str(frame.dtype),
        'temp_dir': user_temp_dir,
        'video_name': video_name,
        'video_path': video_path
    }
    
    # Get video-specific settings
    print(f"🎬 Video path: '{video}' -> Video name: '{video_name}'")
    grid_size_val, vo_points_val, fps_val = get_video_settings(video_name)
    print(f"🎬 Video settings for '{video_name}': grid_size={grid_size_val}, vo_points={vo_points_val}, fps={fps_val}")

    return (json.dumps(frame_data), frame, [], 
            gr.update(value=grid_size_val), 
            gr.update(value=vo_points_val), 
            gr.update(value=fps_val))

def save_masks(o_masks, video_name, temp_dir):
    """Save binary masks to files in user-specific temp directory"""
    o_files = []
    for mask, _ in o_masks:
        o_mask = np.uint8(mask.squeeze() * 255)
        o_file = os.path.join(temp_dir, f"{video_name}.png")
        cv2.imwrite(o_file, o_mask)
        o_files.append(o_file)
    return o_files

def select_point(original_img: str, sel_pix: list, point_type: str, evt: gr.SelectData):
    """Handle point selection for SAM"""
    if original_img is None:
        return None, []
    
    try:
        # Convert stored image data back to numpy array
        frame_data = json.loads(original_img)
        original_img_array = base64_to_numpy(frame_data['data'], frame_data['shape'], frame_data['dtype'])
        temp_dir = frame_data.get('temp_dir', 'temp_local')
        video_name = frame_data.get('video_name', 'video')
        
        # Create a display image for visualization
        display_img = original_img_array.copy()
        new_sel_pix = sel_pix.copy() if sel_pix else []
        new_sel_pix.append((evt.index, 1 if point_type == 'positive_point' else 0))
        
        print(f"🎯 Running SAM inference for point: {evt.index}, type: {point_type}")
        # Run SAM inference
        o_masks = gpu_run_inference(None, original_img_array, new_sel_pix, [])
        
        # Draw points on display image
        for point, label in new_sel_pix:
            cv2.drawMarker(display_img, point, COLORS[label], markerType=MARKERS[label], markerSize=MARKER_SIZE, thickness=2)
        
        # Draw mask overlay on display image
        if o_masks:
            mask = o_masks[0][0]
            overlay = display_img.copy()
            overlay[mask.squeeze()!=0] = [20, 60, 200]  # Light blue
            display_img = cv2.addWeighted(overlay, 0.6, display_img, 0.4, 0)
            
            # Save mask for tracking
            save_masks(o_masks, video_name, temp_dir)
            print(f"✅ Mask saved for video: {video_name}")
        
        return display_img, new_sel_pix
        
    except Exception as e:
        print(f"❌ Error in select_point: {e}")
        return None, []

def reset_points(original_img: str, sel_pix):
    """Reset all points and clear the mask"""
    if original_img is None:
        return None, []
    
    try:
        # Convert stored image data back to numpy array
        frame_data = json.loads(original_img)
        original_img_array = base64_to_numpy(frame_data['data'], frame_data['shape'], frame_data['dtype'])
        temp_dir = frame_data.get('temp_dir', 'temp_local')
        
        # Create a display image (just the original image)
        display_img = original_img_array.copy()
        
        # Clear all points
        new_sel_pix = []
        
        # Clear any existing masks
        for mask_file in glob.glob(os.path.join(temp_dir, "*.png")):
            try:
                os.remove(mask_file)
            except Exception as e:
                logger.warning(f"Failed to remove mask file {mask_file}: {e}")
        
        print("🔄 Points and masks reset")
        return display_img, new_sel_pix
        
    except Exception as e:
        print(f"❌ Error in reset_points: {e}")
        return None, []

def launch_viz(grid_size, vo_points, fps, original_image_state, processing_mode):
    """Launch visualization with user-specific temp directory"""
    if original_image_state is None:
        return None, None, None
    
    try:
        # Get user's temp directory from stored frame data
        frame_data = json.loads(original_image_state)
        temp_dir = frame_data.get('temp_dir', 'temp_local')
        video_name = frame_data.get('video_name', 'video')
        
        print(f"🚀 Starting tracking for video: {video_name}")
        print(f"📊 Parameters: grid_size={grid_size}, vo_points={vo_points}, fps={fps}, mode={processing_mode}")
        
        # Check for mask files
        mask_files = glob.glob(os.path.join(temp_dir, "*.png"))
        video_files = glob.glob(os.path.join(temp_dir, "*.mp4"))
        
        if not video_files:
            print("❌ No video file found")
            return "❌ Error: No video file found", None, None
        
        video_path = video_files[0]
        mask_path = mask_files[0] if mask_files else None
        
        # Run tracker
        print(f"🎯 Running tracker in {processing_mode} mode...")
        out_dir = os.path.join(temp_dir, "results")
        os.makedirs(out_dir, exist_ok=True)

        result_filename = gpu_run_tracker(None, None, temp_dir, video_name, grid_size, vo_points, fps, mode=processing_mode)

        # Process results
        npz_path = os.path.join(out_dir, result_filename)
        track2d_video = os.path.join(out_dir, f"{video_name}_pred_track.mp4")
        
        if os.path.exists(npz_path):
            print("📊 Processing 3D visualization...")
            html_path = process_point_cloud_data(npz_path)
            
            # Schedule deletion of generated files
            delete_later(html_path, delay=600)
            if os.path.exists(track2d_video):
                delete_later(track2d_video, delay=600)
            delete_later(npz_path, delay=600)
            
            # Create iframe HTML
            iframe_html = f"""
            <div style='border: 3px solid #667eea; border-radius: 10px; 
                        background: #f8f9ff; height: 650px; width: 100%;
                        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
                        margin: 0; padding: 0; box-sizing: border-box; overflow: hidden;'>
                <iframe id="viz_iframe" src="/gradio_api/file={html_path}" 
                        width="100%" height="650" frameborder="0" 
                        style="border: none; display: block; width: 100%; height: 650px;
                               margin: 0; padding: 0; border-radius: 7px;">
                </iframe>
            </div>
            """
            
            print("✅ Tracking completed successfully!")
            return iframe_html, track2d_video if os.path.exists(track2d_video) else None, html_path
        else:
            print("❌ Tracking failed - no results generated")
            return "❌ Error: Tracking failed to generate results", None, None
            
    except Exception as e:
        print(f"❌ Error in launch_viz: {e}")
        return f"❌ Error: {str(e)}", None, None

def clear_all():
    """Clear all buffers and temporary files"""
    return (None, None, [], 
            gr.update(value=50), 
            gr.update(value=756), 
            gr.update(value=3))

def clear_all_with_download():
    """Clear all buffers including both download components"""
    return (None, None, [], 
            gr.update(value=50), 
            gr.update(value=756), 
            gr.update(value=3),
            gr.update(value="offline"),  # processing_mode
            None,  # tracking_video_download
            None)  # HTML download component

def get_video_settings(video_name):
    """Get video-specific settings based on video name"""
    video_settings = {
        "running": (50, 512, 2),
        "backpack": (40, 600, 2),
        "kitchen": (60, 800, 3),
        "pillow": (35, 500, 2),
        "handwave": (35, 500, 8),
        "hockey": (45, 700, 2),
        "drifting": (35, 1000, 6),
        "basketball": (45, 1500, 5),
        "ego_teaser": (45, 1200, 10),
        "robot_unitree": (45, 500, 4),
        "robot_3": (35, 400, 5),
        "teleop2": (45, 256, 7),
        "pusht": (45, 256, 10),
        "cinema_0": (45, 356, 5),
        "cinema_1": (45, 756, 3),
        "robot1": (45, 600, 2),
        "robot2": (45, 600, 2),
        "protein": (45, 600, 2),
        "kitchen_egocentric": (45, 600, 2),
        "ball_ke": (50, 600, 3), 
        "groundbox_800": (50, 756, 3),
        "mug": (50, 756, 3), 
    }
    
    return video_settings.get(video_name, (50, 756, 3)) 

def update_status_indicator(processing_mode):
    """Update status indicator based on processing mode"""
    if processing_mode == "offline":
        return "**Status:** 🟢 Local Processing Mode (Offline)"
    else:
        return "**Status:** 🔵 Cloud Processing Mode (Online)"

# Create the Gradio interface
print("🎨 Creating Gradio interface...")

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="🎯 [SpatialTracker V2](https://github.com/henry123-boy/SpaTrackerV2)",
    css="""
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
    }
    .gr-button {
        margin: 5px;
    }
    .gr-form {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    /* 移除 gr.Group 的默认灰色背景 */
    .gr-form {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }
    /* 固定3D可视化器尺寸 */
    #viz_container {
        height: 650px !important;
        min-height: 650px !important;
        max-height: 650px !important;
        width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: hidden !important;
    }
    #viz_container > div {
        height: 650px !important;
        min-height: 650px !important;
        max-height: 650px !important;
        width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
        box-sizing: border-box !important;
    }
    #viz_container iframe {
        height: 650px !important;
        min-height: 650px !important;
        max-height: 650px !important;
        width: 100% !important;
        border: none !important;
        display: block !important;
        margin: 0 !important;
        padding: 0 !important;
        box-sizing: border-box !important;
    }
    /* 固定视频上传组件高度 */
    .gr-video {
        height: 300px !important;
        min-height: 300px !important;
        max-height: 300px !important;
    }
    .gr-video video {
        height: 260px !important;
        max-height: 260px !important;
        object-fit: contain !important;
        background: #f8f9fa;
    }
    .gr-video .gr-video-player {
        height: 260px !important;
        max-height: 260px !important;
    }
    /* 强力移除examples的灰色背景 - 使用更通用的选择器 */
    .horizontal-examples,
    .horizontal-examples > *,
    .horizontal-examples * {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
    }
    
    /* Examples组件水平滚动样式 */
    .horizontal-examples [data-testid="examples"] {
        background: transparent !important;
        background-color: transparent !important;
    }
    
    .horizontal-examples [data-testid="examples"] > div {
        background: transparent !important;
        background-color: transparent !important;
        overflow-x: auto !important;
        overflow-y: hidden !important;
        scrollbar-width: thin;
        scrollbar-color: #667eea transparent;
        padding: 0 !important;
        margin-top: 10px;
        border: none !important;
    }
    
    .horizontal-examples [data-testid="examples"] table {
        display: flex !important;
        flex-wrap: nowrap !important;
        min-width: max-content !important;
        gap: 15px !important;
        padding: 10px 0;
        background: transparent !important;
        border: none !important;
    }
    
    .horizontal-examples [data-testid="examples"] tbody {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        gap: 15px !important;
        background: transparent !important;
    }
    
    .horizontal-examples [data-testid="examples"] tr {
        display: flex !important;
        flex-direction: column !important;
        min-width: 160px !important;
        max-width: 160px !important;
        margin: 0 !important;
        background: white !important;
        border-radius: 12px;
        box-shadow: 0 3px 12px rgba(0,0,0,0.12);
        transition: all 0.3s ease;
        cursor: pointer;
        overflow: hidden;
        border: none !important;
    }
    
    .horizontal-examples [data-testid="examples"] tr:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.25);
    }
    
    .horizontal-examples [data-testid="examples"] td {
        text-align: center !important;
        padding: 0 !important;
        border: none !important;
        background: transparent !important;
    }
    
    .horizontal-examples [data-testid="examples"] td:first-child {
        padding: 0 !important;
        background: transparent !important;
    }
    
    .horizontal-examples [data-testid="examples"] video {
        border-radius: 8px 8px 0 0 !important;
        width: 100% !important;
        height: 90px !important;
        object-fit: cover !important;
        background: #f8f9fa !important;
    }
    
    .horizontal-examples [data-testid="examples"] td:last-child {
        font-size: 11px !important;
        font-weight: 600 !important;
        color: #333 !important;
        padding: 8px 12px !important;
        background: linear-gradient(135deg, #f8f9ff 0%, #e6f3ff 100%) !important;
        border-radius: 0 0 8px 8px;
    }
    
    /* 滚动条样式 */
    .horizontal-examples [data-testid="examples"] > div::-webkit-scrollbar {
        height: 8px;
    }
    .horizontal-examples [data-testid="examples"] > div::-webkit-scrollbar-track {
        background: transparent;
        border-radius: 4px;
    }
    .horizontal-examples [data-testid="examples"] > div::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
    }
    .horizontal-examples [data-testid="examples"] > div::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%);
    }
    """
) as demo:
    
    # Add prominent main title
    
    gr.Markdown("""
    # ✨ SpatialTrackerV2
                
    Welcome to [SpatialTracker V2](https://github.com/henry123-boy/SpaTrackerV2)! This interface allows you to track any pixels in 3D using our model.
    For full information, please refer to the [official website](https://spatialtracker.github.io/), and [ICCV2025 paper](https://github.com/henry123-boy/SpaTrackerV2).
    Please cite our paper and give us a star 🌟 if you find this project useful!
    
    **⚡ Quick Start:** Upload video → Click "Start Tracking Now!"
    
    **🔬 Advanced Usage with SAM:**
    1. Upload a video file or select from examples below
    2. Expand "Manual Point Selection" to click on specific objects for SAM-guided tracking
    3. Adjust tracking parameters for optimal performance  
    4. Click "Start Tracking Now!" to begin 3D tracking with SAM guidance
    
    """)
    
    # Status indicator
    status_indicator = gr.Markdown("**Status:** 🟢 Local Processing Mode (Offline)")
    
    # Main content area - video upload left, 3D visualization right
    with gr.Row():
        with gr.Column(scale=1):
            # Video upload section
            gr.Markdown("### 📂 Select Video")
            
            # Define video_input here so it can be referenced in examples
            video_input = gr.Video(
                label="Upload Video or Select Example",
                format="mp4",
                height=250  # Matched height with 3D viz
            )
                

            # Traditional examples but with horizontal scroll styling
            gr.Markdown("🎨**Examples:** (scroll horizontally to see all videos)")
            with gr.Row(elem_classes=["horizontal-examples"]):
                # Horizontal video examples with slider
                # gr.HTML("<div style='margin-top: 5px;'></div>")
                gr.Examples(
                    examples=[
                        ["./examples/robot1.mp4"],
                        ["./examples/robot2.mp4"],
                        ["./examples/protein.mp4"],
                        ["./examples/groundbox_800.mp4"],
                        ["./examples/kitchen_egocentric.mp4"],
                        ["./examples/hockey.mp4"],
                        ["./examples/running.mp4"],
                        ["./examples/ball_ke.mp4"],
                        ["./examples/mug.mp4"],
                        ["./examples/robot_3.mp4"],
                        ["./examples/backpack.mp4"],
                        ["./examples/kitchen.mp4"],
                        ["./examples/pillow.mp4"],
                        ["./examples/handwave.mp4"],
                        ["./examples/drifting.mp4"],
                        ["./examples/basketball.mp4"],
                        ["./examples/ken_block_0.mp4"],
                        ["./examples/ego_kc1.mp4"],
                        ["./examples/vertical_place.mp4"],
                        ["./examples/ego_teaser.mp4"],
                        ["./examples/robot_unitree.mp4"],
                        ["./examples/teleop2.mp4"],
                        ["./examples/pusht.mp4"],
                        ["./examples/cinema_0.mp4"],
                        ["./examples/cinema_1.mp4"],
                    ],
                    inputs=[video_input],
                    outputs=[video_input],
                    fn=None,
                    cache_examples=False,
                    label="",
                    examples_per_page=6  # Show 6 examples per page so they can wrap to multiple rows
                )
        
        with gr.Column(scale=2):
            # 3D Visualization - wider and taller to match left side
            with gr.Group():
                gr.Markdown("### 🌐 3D Trajectory Visualization")
                viz_html = gr.HTML(
                    label="3D Trajectory Visualization",
                    value="""
                    <div style='border: 3px solid #667eea; border-radius: 10px; 
                                background: linear-gradient(135deg, #f8f9ff 0%, #e6f3ff 100%); 
                                text-align: center; height: 650px; display: flex; 
                                flex-direction: column; justify-content: center; align-items: center;
                                box-shadow: 0 4px 16px rgba(102, 126, 234, 0.15);
                                margin: 0; padding: 20px; box-sizing: border-box;'>
                        <div style='font-size: 56px; margin-bottom: 25px;'>🌐</div>
                        <h3 style='color: #667eea; margin-bottom: 18px; font-size: 28px; font-weight: 600;'>
                            3D Trajectory Visualization
                        </h3>
                        <p style='color: #666; font-size: 18px; line-height: 1.6; max-width: 550px; margin-bottom: 30px;'>
                            Track any pixels in 3D space with camera motion
                        </p>
                        <div style='background: rgba(102, 126, 234, 0.1); border-radius: 30px; 
                                    padding: 15px 30px; border: 1px solid rgba(102, 126, 234, 0.2);'>
                            <span style='color: #667eea; font-weight: 600; font-size: 16px;'>
                                ⚡ Powered by SpatialTracker V2
                            </span>
                        </div>
                    </div>
                    """,
                    elem_id="viz_container"
                )

    # Start button section - below video area
    with gr.Row():
        with gr.Column(scale=3):
            launch_btn = gr.Button("🚀 Start Tracking Now!", variant="primary", size="lg")
        with gr.Column(scale=1):
            clear_all_btn = gr.Button("🗑️ Clear All", variant="secondary", size="sm")

    # Tracking parameters section
    with gr.Row():
        gr.Markdown("### ⚙️ Tracking Parameters")
    with gr.Row():
        # 添加模式选择器
        with gr.Column(scale=1):
            processing_mode = gr.Radio(
                choices=["offline", "online"],
                value="offline",
                label="Processing Mode",
                info="Offline: default mode | Online: Sliding Window Mode"
            )
        with gr.Column(scale=1):
            grid_size = gr.Slider(
                minimum=10, maximum=100, step=10, value=50,
                label="Grid Size", info="Tracking detail level"
            )
        with gr.Column(scale=1):
            vo_points = gr.Slider(
                minimum=100, maximum=2000, step=50, value=756,
                label="VO Points", info="Motion accuracy"
            )
        with gr.Column(scale=1):
            fps = gr.Slider(
                minimum=1, maximum=20, step=1, value=3,
                label="FPS", info="Processing speed"
            )

    # Advanced Point Selection with SAM - Collapsed by default
    with gr.Row():
        gr.Markdown("### 🎯 Advanced: Manual Point Selection with SAM")
    with gr.Accordion("🔬 SAM Point Selection Controls", open=False):
        gr.HTML("""
        <div style='margin-bottom: 15px;'>
            <ul style='color: #4a5568; font-size: 14px; line-height: 1.6; margin: 0; padding-left: 20px;'>
                <li>Click on target objects in the image for SAM-guided segmentation</li>
                <li>Positive points: include these areas | Negative points: exclude these areas</li>
                <li>Get more accurate 3D tracking results with SAM's powerful segmentation</li>
            </ul>
        </div>
        """)
        
        with gr.Row():
            with gr.Column():
                interactive_frame = gr.Image(
                    label="Click to select tracking points with SAM guidance",
                    type="numpy",
                    interactive=True,
                    height=300
                )
                
                with gr.Row():
                    point_type = gr.Radio(
                        choices=["positive_point", "negative_point"],
                        value="positive_point",
                        label="Point Type",
                        info="Positive: track these areas | Negative: avoid these areas"
                    )
                    
                with gr.Row():
                    reset_points_btn = gr.Button("🔄 Reset Points", variant="secondary", size="sm")

    # Downloads section - hidden but still functional for local processing
    with gr.Row(visible=False):
        with gr.Column(scale=1):
            tracking_video_download = gr.File(
                label="📹 Download 2D Tracking Video",
                interactive=False,
                visible=False
            )
        with gr.Column(scale=1):
            html_download = gr.File(
                label="📄 Download 3D Visualization HTML",
                interactive=False,
                visible=False
            )

    # GitHub Star Section
    gr.HTML("""
    <div style='background: linear-gradient(135deg, #e8eaff 0%, #f0f2ff 100%); 
                border-radius: 8px; padding: 20px; margin: 15px 0; 
                box-shadow: 0 2px 8px rgba(102, 126, 234, 0.1);
                border: 1px solid rgba(102, 126, 234, 0.15);'>
        <div style='text-align: center;'>
            <h3 style='color: #4a5568; margin: 0 0 10px 0; font-size: 18px; font-weight: 600;'>
                ⭐ Love SpatialTracker? Give us a Star! ⭐
            </h3>
            <p style='color: #666; margin: 0 0 15px 0; font-size: 14px; line-height: 1.5;'>
                Help us grow by starring our repository on GitHub! Your support means a lot to the community. 🚀
            </p>
            <a href="https://github.com/henry123-boy/SpaTrackerV2" target="_blank" 
               style='display: inline-flex; align-items: center; gap: 8px; 
                      background: rgba(102, 126, 234, 0.1); color: #4a5568; 
                      padding: 10px 20px; border-radius: 25px; text-decoration: none; 
                      font-weight: bold; font-size: 14px; border: 1px solid rgba(102, 126, 234, 0.2);
                      transition: all 0.3s ease;'
               onmouseover="this.style.background='rgba(102, 126, 234, 0.15)'; this.style.transform='translateY(-2px)'"
               onmouseout="this.style.background='rgba(102, 126, 234, 0.1)'; this.style.transform='translateY(0)'">
                <span style='font-size: 16px;'>⭐</span>
                Star SpatialTracker V2 on GitHub
            </a>
        </div>
    </div>
    """)
    
    # Acknowledgments Section
    gr.HTML("""
    <div style='background: linear-gradient(135deg, #fff8e1 0%, #fffbf0 100%); 
                border-radius: 8px; padding: 20px; margin: 15px 0; 
                box-shadow: 0 2px 8px rgba(255, 193, 7, 0.1);
                border: 1px solid rgba(255, 193, 7, 0.2);'>
        <div style='text-align: center;'>
            <h3 style='color: #5d4037; margin: 0 0 10px 0; font-size: 18px; font-weight: 600;'>
                📚 Acknowledgments
            </h3>
            <p style='color: #5d4037; margin: 0 0 15px 0; font-size: 14px; line-height: 1.5;'>
                Our 3D visualizer is adapted from <strong>TAPIP3D</strong>. We thank the authors for their excellent work and contribution to the computer vision community!
            </p>
            <a href="https://github.com/zbw001/TAPIP3D" target="_blank" 
               style='display: inline-flex; align-items: center; gap: 8px; 
                      background: rgba(255, 193, 7, 0.15); color: #5d4037; 
                      padding: 10px 20px; border-radius: 25px; text-decoration: none; 
                      font-weight: bold; font-size: 14px; border: 1px solid rgba(255, 193, 7, 0.3);
                      transition: all 0.3s ease;'
               onmouseover="this.style.background='rgba(255, 193, 7, 0.25)'; this.style.transform='translateY(-2px)'"
               onmouseout="this.style.background='rgba(255, 193, 7, 0.15)'; this.style.transform='translateY(0)'">
                📚 Visit TAPIP3D Repository
            </a>
        </div>
    </div>
    """)
    
    # Footer
    gr.HTML("""
    <div style='text-align: center; margin: 20px 0 10px 0;'>
        <span style='font-size: 12px; color: #888; font-style: italic;'>
            Powered by SpatialTracker V2 | Built with ❤️ for the Computer Vision Community
        </span>
    </div>
    """)

    # Hidden state variables
    original_image_state = gr.State(None)
    selected_points = gr.State([])
    
    # Event handlers
    video_input.change(
        fn=handle_video_upload,
        inputs=[video_input],
        outputs=[original_image_state, interactive_frame, selected_points, grid_size, vo_points, fps]
    )
    
    processing_mode.change(
        fn=update_status_indicator,
        inputs=[processing_mode],
        outputs=[status_indicator]
    )
    
    interactive_frame.select(
        fn=select_point,
        inputs=[original_image_state, selected_points, point_type],
        outputs=[interactive_frame, selected_points]
    )
    
    reset_points_btn.click(
        fn=reset_points,
        inputs=[original_image_state, selected_points],
        outputs=[interactive_frame, selected_points]
    )
    
    clear_all_btn.click(
        fn=clear_all_with_download,
        outputs=[video_input, interactive_frame, selected_points, grid_size, vo_points, fps, processing_mode, tracking_video_download, html_download]
    )
    
    launch_btn.click(
        fn=launch_viz,
        inputs=[grid_size, vo_points, fps, original_image_state, processing_mode],
        outputs=[viz_html, tracking_video_download, html_download]
    )

# Launch the interface
if __name__ == "__main__":
    print("🌟 Launching SpatialTracker V2 Local Version...")
    print("🔗 Running in Local Processing Mode")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True
    ) 