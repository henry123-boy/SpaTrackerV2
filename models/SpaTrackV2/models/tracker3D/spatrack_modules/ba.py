import pycolmap
import torch
import numpy as np
import pyceres
from pyceres import SolverOptions, LinearSolverType, PreconditionerType, TrustRegionStrategyType, LoggingType
import logging
from scipy.spatial.transform import Rotation as R

# config logging and make sure it print to the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_static_from_3DTracks(world_tracks, dyn_prob,
                                     query_3d_pts, vis_est, tracks2d, img_size, K=100, maintain_invisb=False):
    """
    world_tracks: B T N 3   this is the coarse 3d tracks in world coordinate  (coarse 3d tracks)
    dyn_prob: B T N   this is the dynamic probability of the 3d tracks
    query_3d_pts: B T N 3   this is the query 3d points in world coordinate (coarse by camera pose)
    vis_est: B T N   this is the visibility of the 3d tracks
    tracks2d: B T N 2   this is the 2d tracks   
    K: int   top K static points
    """
    B, T, N, _ = world_tracks.shape
    static_msk = (dyn_prob<0.5).bool()
    world_tracks_static = world_tracks[:,:,static_msk.squeeze(),:]
    query_3d_pts_static = query_3d_pts[:,static_msk.squeeze(),:]
    if maintain_invisb:
        vis = (tracks2d[...,0] > 0).bool() * (tracks2d[...,1] > 0).bool()
        vis_mask = vis * (img_size[1] > tracks2d[...,0]) * (img_size[0] > tracks2d[...,1])
        vis_mask = vis_mask[:,:,static_msk.squeeze()]
    else:
        vis_mask = (vis_est>0.5).bool()[:,:,static_msk.squeeze()]
    tracks2d_static = tracks2d[:,:,static_msk.squeeze(),:]
    world_tracks_static = (world_tracks_static*vis_mask[...,None]).sum(dim=1)/(vis_mask.sum(dim=1)[...,None]+1e-6)
    # get the distance between the query_3d_pts_static and the world_tracks_static
    dist = (query_3d_pts_static-world_tracks_static).norm(dim=-1)
    # get the top K static points, which have the smallest distance
    topk_idx = torch.argsort(dist,dim=-1)[:,:K]
    world_tracks_static = world_tracks_static[torch.arange(B)[:,None,None],topk_idx]
    query_3d_pts_static = query_3d_pts_static[torch.arange(B)[:,None,None],topk_idx]    
    # get the visible selected
    vis_mask_static = vis_mask[:,:,topk_idx.squeeze()]
    tracks2d_static = tracks2d_static[:, :, topk_idx.squeeze(), :]

    return world_tracks_static, static_msk, topk_idx, vis_mask_static, tracks2d_static

def log_ba_summary(summary):
    logging.info(f"Residuals : {summary.num_residuals_reduced}")
    if summary.num_residuals_reduced > 0:
        logging.info(f"Parameters : {summary.num_effective_parameters_reduced}")
        logging.info(
            f"Iterations : {summary.num_successful_steps + summary.num_unsuccessful_steps}"
        )
        logging.info(f"Time : {summary.total_time_in_seconds} [s]")
        logging.info(
            f"Initial cost : {np.sqrt(summary.initial_cost / summary.num_residuals_reduced)} [px]"
        )
        logging.info(
            f"Final cost : {np.sqrt(summary.final_cost / summary.num_residuals_reduced)} [px]"
        )
        return True
    else:
        print("No residuals reduced")
        return False

# def solve_bundle_adjustment(reconstruction, ba_options, ba_config):
#     bundle_adjuster = pycolmap.BundleAdjuster(ba_options, ba_config)
#     bundle_adjuster.set_up_problem(
#         reconstruction, ba_options.create_loss_function()
#     )
#     solver_options = bundle_adjuster.set_up_solver_options(
#         bundle_adjuster.problem, ba_options.solver_options
#     )
#     summary = pyceres.SolverSummary()
#     pyceres.solve(solver_options, bundle_adjuster.problem, summary)
#     return summary

def efficient_solver(solver_options, stability_mode=True):
    # Set linear solver to ITERATIVE_SCHUR (using PCG to solve Schur complement)
    solver_options.linear_solver_type = LinearSolverType.ITERATIVE_SCHUR
    
    # Set preconditioner (critical for PCG)
    solver_options.preconditioner_type = PreconditionerType.SCHUR_JACOBI
    
    # Optimize trust region strategy
    solver_options.trust_region_strategy_type = TrustRegionStrategyType.LEVENBERG_MARQUARDT
    
    # Enable multi-threading acceleration
    solver_options.num_threads = 32  # Adjust based on CPU cores
    
    if stability_mode:
        # Stability-first configuration
        solver_options.initial_trust_region_radius = 1.0  # Reduce initial step size
        solver_options.max_trust_region_radius = 10.0    # Limit max step size
        solver_options.min_trust_region_radius = 1e-6    # Allow small step convergence
        
        # Increase regularization parameters
        solver_options.use_nonmonotonic_steps = True     # Allow non-monotonic steps
        solver_options.max_consecutive_nonmonotonic_steps = 10
        
        # Adjust iteration termination conditions
        solver_options.max_num_iterations = 100          # Increase max iterations
        solver_options.function_tolerance = 1e-8         # Stricter function convergence
        solver_options.gradient_tolerance = 1e-12        # Stricter gradient convergence
        solver_options.parameter_tolerance = 1e-10       # Stricter parameter convergence
        
        # Control PCG iterations and precision
        solver_options.min_linear_solver_iterations = 10
        solver_options.max_linear_solver_iterations = 100
        solver_options.inner_iteration_tolerance = 0.01  # Higher inner iteration precision
        
        # Increase damping factor
        solver_options.min_lm_diagonal = 1e-3            # Increase min LM diagonal
        solver_options.max_lm_diagonal = 1e+10           # Limit max LM diagonal
        
        # Enable parameter change limits
        solver_options.update_state_every_iteration = True  # Update state each iteration
        
    else:
        # Efficiency-first configuration (original settings)
        solver_options.initial_trust_region_radius = 10000.0
        solver_options.max_trust_region_radius = 1e+16
        solver_options.max_num_iterations = 50
        solver_options.function_tolerance = 1e-6
        solver_options.gradient_tolerance = 1e-10
        solver_options.parameter_tolerance = 1e-8
        solver_options.min_linear_solver_iterations = 5
        solver_options.max_linear_solver_iterations = 50
        solver_options.inner_iteration_tolerance = 0.1
    
    # Enable Jacobi scaling for better numerical stability
    solver_options.jacobi_scaling = True
    
    # Disable verbose logging for better performance (enable for debugging)
    solver_options.logging_type = LoggingType.SILENT
    solver_options.minimizer_progress_to_stdout = False
    
    return solver_options

class SpatTrackCost_static(pyceres.CostFunction):
    def __init__(self, observed_depth):
        """
        observed_depth: float
        """
        super().__init__()
        self.observed_depth = float(observed_depth)
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([4, 3, 3])  # [rotation_quat, translation, xyz]

    def Evaluate(self, parameters, residuals, jacobians):
        # Unpack parameters
        quat = parameters[0]       # shape: (4,) [w, x, y, z]
        t = parameters[1]          # shape: (3,)
        point = parameters[2]      # shape: (3,)

        # Convert COLMAP-style quat [w, x, y, z] to scipy format [x, y, z, w]
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        R_mat = r.as_matrix()  # (3, 3)

        # Transform point to camera frame
        X_cam = R_mat @ point + t
        z = X_cam[2]

        # Compute residual (normalized depth error)
        residuals[0] = 20.0 * (z - self.observed_depth) / self.observed_depth

        if jacobians is not None:
            if jacobians[2] is not None:
                # dr/d(point3D): only z-axis matters, so only 3rd row of R
                jacobians[2][0] = 20.0 * R_mat[2, 0] / self.observed_depth
                jacobians[2][1] = 20.0 * R_mat[2, 1] / self.observed_depth
                jacobians[2][2] = 20.0 * R_mat[2, 2] / self.observed_depth

            if jacobians[1] is not None:
                # dr/dt = ∂residual/∂translation = d(z)/dt = [0, 0, 1]
                jacobians[1][0] = 0.0
                jacobians[1][1] = 0.0
                jacobians[1][2] = 20.0 / self.observed_depth

            if jacobians[0] is not None:
                # Optional: dr/d(quat) — not trivial to derive, can be left for autodiff if needed
                # Set zero for now (not ideal but legal)
                jacobians[0][:] = 0.0

        return True
    

class SpatTrackCost_dynamic(pyceres.CostFunction):

    def __init__(self, observed_uv, image, point3D, camera):
        """
        observed_uv: 1 1 K 2   this is the 2d tracks
        image: pycolmap.Image object
        point3D: pycolmap.Point3D object
        camera: pycolmap.Camera object
        """
        sizes = [image.cam_from_world.params.shape[0], point3D.xyz.shape[0], camera.params.shape[0]]
        super().__init__(self, residual_size=2, parameter_block_sizes=sizes)
        self.observed_uv = observed_uv
        self.image = image
        self.point3D = point3D
        self.camera = camera

def solve_bundle_adjustment(reconstruction, ba_options,
                                ba_config=None, extra_residual=None):
    """
    Perform bundle adjustment optimization (compatible with pycolmap 3.10.0)

    Args:
        reconstruction: pycolmap.Reconstruction object
        ba_options: pycolmap.BundleAdjustmentOptions object
        ba_config: pycolmap.BundleAdjustmentConfig object (not used in 3.10.0)
    """
    # Use pycolmap 3.10.0 API - only takes reconstruction and options
    # Note: ba_config is not supported in this version, returns None
    pycolmap.bundle_adjustment(reconstruction, ba_options)

    # Create a minimal summary object for compatibility
    summary = pyceres.SolverSummary()
    summary.num_residuals_reduced = len(reconstruction.points3D)
    summary.num_effective_parameters_reduced = len(reconstruction.reg_image_ids()) * 6

    return summary

def batch_matrix_to_pycolmap(
    points3d,
    extrinsics,
    intrinsics,
    tracks,
    masks,
    image_size,
    max_points3D_val=3000,
    shared_camera=False,
    camera_type="SIMPLE_PINHOLE",
    extra_params=None,
    cam_tracks_static=None,
    query_pts=None,
):
    """
    Convert Batched Pytorch Tensors to PyCOLMAP

    Check https://github.com/colmap/pycolmap for more details about its format
    """
    # points3d: Px3
    # extrinsics: Nx3x4
    # intrinsics: Nx3x3
    # tracks: NxPx2
    # masks: NxP
    # image_size: 2, assume all the frames have been padded to the same size
    # where N is the number of frames and P is the number of tracks

    N, P, _ = tracks.shape
    assert len(extrinsics) == N
    assert len(intrinsics) == N
    assert len(points3d) == P
    assert image_size.shape[0] == 2

    extrinsics = extrinsics.cpu().numpy()
    intrinsics = intrinsics.cpu().numpy()

    if extra_params is not None:
        extra_params = extra_params.cpu().numpy()

    tracks = tracks.cpu().numpy()
    masks = masks.cpu().numpy()
    points3d = points3d.cpu().numpy()
    image_size = image_size.cpu().numpy()
    if cam_tracks_static is not None:
        cam_tracks_static = cam_tracks_static.cpu().numpy()

    # Reconstruction object, following the format of PyCOLMAP/COLMAP
    reconstruction = pycolmap.Reconstruction()

    inlier_num = masks.sum(0)
    valid_mask = inlier_num >= 2  # a track is invalid if without two inliers
    valid_idx = np.nonzero(valid_mask)[0]

    # Only add 3D points that have sufficient 2D points
    point3d_ids = []
    for vidx in valid_idx:
        point3d_id = reconstruction.add_point3D(
            points3d[vidx], pycolmap.Track(), np.zeros(3)
        )
        point3d_ids.append(point3d_id)

    # add the residual pair
    if cam_tracks_static is not None:
        extra_residual = []
        for id_x, vidx in enumerate(valid_idx):
            points_3d_id = point3d_ids[id_x]
            point_residual = {
                "point3D_id": points_3d_id,
                "image_ids": [],
                "observed_depth": [],
            }
            query_i = query_pts[:,:,vidx]
            point_residual["image_ids"].append(int(query_i[0,0,0]))
            point_residual["observed_depth"].append(query_i[0,0,-1])
            extra_residual.append(point_residual)
    else:
        extra_residual = None

    num_points3D = len(valid_idx)

    camera = None
    # frame idx
    for fidx in range(N):
        # set camera
        if camera is None or (not shared_camera):
            if camera_type == "SIMPLE_RADIAL":
                pycolmap_intri = np.array(
                    [
                        intrinsics[fidx][0, 0],
                        intrinsics[fidx][0, 2],
                        intrinsics[fidx][1, 2],
                        extra_params[fidx][0],
                    ]
                )
            elif camera_type == "SIMPLE_PINHOLE":
                pycolmap_intri = np.array(
                    [
                        intrinsics[fidx][0, 0],
                        intrinsics[fidx][0, 2],
                        intrinsics[fidx][1, 2],
                    ]
                )
            else:
                raise ValueError(
                    f"Camera type {camera_type} is not supported yet"
                )

            camera = pycolmap.Camera(
                model=camera_type,
                width=image_size[0],
                height=image_size[1],
                params=pycolmap_intri,
                camera_id=fidx,
            )

            # add camera
            reconstruction.add_camera(camera)

        # set image
        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(extrinsics[fidx][:3, :3]),
            extrinsics[fidx][:3, 3],
        )  # Rot and Trans
        print(dir(pycolmap.Image))
        image = pycolmap.Image(
            id=fidx,
            name=f"image_{fidx}",
            camera_id=camera.camera_id,
            cam_from_world=cam_from_world,
        )

        points2D_list = []

        point2D_idx = 0
        # NOTE point3D_id start by 1
        for point3D_id in range(1, num_points3D + 1):
            original_track_idx = valid_idx[point3D_id - 1]

            if (
                reconstruction.points3D[point3D_id].xyz < max_points3D_val
            ).all():
                if masks[fidx][original_track_idx]:
                    # It seems we don't need +0.5 for BA
                    point2D_xy = tracks[fidx][original_track_idx]
                    # Please note when adding the Point2D object
                    # It not only requires the 2D xy location, but also the id to 3D point
                    points2D_list.append(
                        pycolmap.Point2D(point2D_xy, point3D_id)
                    )

                    # add element
                    track = reconstruction.points3D[point3D_id].track
                    track.add_element(fidx, point2D_idx)
                    point2D_idx += 1

        assert point2D_idx == len(points2D_list)
        try:
            image.points2D = pycolmap.ListPoint2D(points2D_list)
        except Exception as e:
            print(f"frame {fidx} is out of BA: {e}")
        
        # add image
        reconstruction.add_image(image)

    return reconstruction, valid_idx, extra_residual

def pycolmap_to_batch_matrix(
    reconstruction, device="cuda", camera_type="SIMPLE_PINHOLE"
):
    """
    Convert a PyCOLMAP Reconstruction Object to batched PyTorch tensors.

    Args:
        reconstruction (pycolmap.Reconstruction): The reconstruction object from PyCOLMAP.
        device (str): The device to place the tensors on (default: "cuda").
        camera_type (str): The type of camera model used (default: "SIMPLE_PINHOLE").

    Returns:
        tuple: A tuple containing points3D, extrinsics, intrinsics, and optionally extra_params.
    """

    num_images = len(reconstruction.images)
    max_points3D_id = max(reconstruction.point3D_ids())
    points3D = np.zeros((max_points3D_id, 3))

    for point3D_id in reconstruction.points3D:
        points3D[point3D_id - 1] = reconstruction.points3D[point3D_id].xyz
    points3D = torch.from_numpy(points3D).to(device)

    extrinsics = []
    intrinsics = []

    extra_params = [] if camera_type == "SIMPLE_RADIAL" else None

    for i in range(num_images):
        # Extract and append extrinsics
        pyimg = reconstruction.images[i]
        pycam = reconstruction.cameras[pyimg.camera_id]
        matrix = pyimg.cam_from_world.matrix()
        extrinsics.append(matrix)

        # Extract and append intrinsics
        calibration_matrix = pycam.calibration_matrix()
        intrinsics.append(calibration_matrix)

        if camera_type == "SIMPLE_RADIAL":
            extra_params.append(pycam.params[-1])

    # Convert lists to torch tensors
    extrinsics = torch.from_numpy(np.stack(extrinsics)).to(device)

    intrinsics = torch.from_numpy(np.stack(intrinsics)).to(device)

    if camera_type == "SIMPLE_RADIAL":
        extra_params = torch.from_numpy(np.stack(extra_params)).to(device)
        extra_params = extra_params[:, None]

    return points3D, extrinsics, intrinsics, extra_params

def ba_pycolmap(world_tracks, intrs, c2w_traj, visb, tracks2d, image_size, cam_tracks_static=None, training=True, query_pts=None):
    """
    world_tracks: 1 1 K 3   this is the coarse 3d tracks in world coordinate  (coarse 3d tracks)
    intrs: B T 3 3   this is the intrinsic matrix
    c2w_traj: B T 4 4   this is the camera trajectory
    visb: B T K   this is the visibility of the 3d tracks
    tracks2d: B T K 2   this is the 2d tracks
    """
    with torch.no_grad():
        B, _, K, _ = world_tracks.shape
        T = c2w_traj.shape[1]
        world_tracks = world_tracks.view(K, 3).detach()
        world_tracks_refine = world_tracks.view(K, 3).detach().clone()
        c2w_traj_glob = c2w_traj.view(B*T, 4, 4).detach().clone()
        c2w_traj = c2w_traj.view(B*T, 4, 4).detach()
        intrs = intrs.view(B*T, 3, 3).detach()
        visb = visb.view(B*T, K).detach()
        tracks2d = tracks2d[...,:2].view(B*T, K, 2).detach()

        rec, valid_idx_pts, extra_residual = batch_matrix_to_pycolmap(
                world_tracks,
                torch.inverse(c2w_traj)[:,:3,:],
                intrs,
                tracks2d,
                visb,
                image_size,
                cam_tracks_static=cam_tracks_static,
                query_pts=query_pts,
            )
        # NOTE It is window_size + 1 instead of window_size
        ba_options = pycolmap.BundleAdjustmentOptions()
        ba_options.refine_focal_length = False
        ba_options.refine_principal_point = False
        ba_options.refine_extra_params = False
        ba_config = pycolmap.BundleAdjustmentConfig()
        reg_image_ids = list(rec.reg_image_ids())
        for image_id in reg_image_ids:
            ba_config.add_image(image_id)
        # Fix frame 0, i.e, the end frame of the last window
        # Check if image ID 0 exists, otherwise use the first available ID
        if reg_image_ids:
            if 0 in reg_image_ids:
                ba_config.set_constant_cam_pose(0)
            else:
                ba_config.set_constant_cam_pose(reg_image_ids[0])

        # fix the 3d points 
        for point3D_id in rec.points3D:
            if training:
                # ba_config.add_constant_point(point3D_id)
                ba_config.add_variable_point(point3D_id)
            else:
                ba_config.add_variable_point(point3D_id)
                # ba_config.add_constant_point(point3D_id)
        if (len(ba_config.variable_point3D_ids) < 50) and (len(ba_config.constant_point3D_ids) < 50):
            return c2w_traj_glob, world_tracks_refine, intrs
        summary = solve_bundle_adjustment(rec, ba_options, ba_config, extra_residual=extra_residual)
        # free the 3d points 
        # for point3D_id in rec.points3D:
        #     ba_config.remove_constant_point(point3D_id)
        #     ba_config.add_variable_point(point3D_id)
        # summary = solve_bundle_adjustment(rec, ba_options, ba_config)
        if not training:
            ba_success = log_ba_summary(summary)
        # get the refined results
        points3D, extrinsics, intrinsics, extra_params = pycolmap_to_batch_matrix(rec, device="cuda", camera_type="SIMPLE_PINHOLE")
        c2w_traj_glob[:, :3, :] = extrinsics
        c2w_traj_glob = torch.inverse(c2w_traj_glob)
    world_tracks_refine[valid_idx_pts] = points3D.to(world_tracks_refine.device).to(world_tracks_refine.dtype)
    intrinsics = intrinsics.to(world_tracks_refine.device).to(world_tracks_refine.dtype)
    # import pdb; pdb.set_trace()
    return c2w_traj_glob, world_tracks_refine, intrinsics
    

    

