import os
import time
import numpy as np
import matplotlib.pyplot as plt

from motrixsim import SceneData, SceneModel, load_model, run, step
from motrixsim.render import Color, RenderApp

from absl import app, flags
from mujoco.mjmx_bridge import MjMxBridge

try:
    import jax
    import jax.numpy as jnp
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3" # 如果显存充足，可以调大一些

    import mujoco_lidar
    from mujoco_lidar import scan_gen
    from mujoco_lidar.core_jax import MjLidarJax
    assert mujoco_lidar.__version__ >= "0.2.3", "Please upgrade mujoco-lidar to version 0.2.3 or higher."

except ImportError:
    print("[ERROR] mujoco_lidar package not found. Please install mujoco-lidar[jax] to run this example.")
    print("Visit https://github.com/TATP-233/MuJoCo-LiDAR for installation instructions.")
    print("\nInstallation steps:")
    print("  git clone https://github.com/TATP-233/MuJoCo-LiDAR.git")
    print("  cd MuJoCo-LiDAR")
    print("  pip install -e \".[jax]\"")
    print("\nVerify JAX installation:")
    print("  python -c \"import jax; print(jax.default_backend())\"")
    print("  (Should print 'gpu')")
    exit(0)

print("=" * 60)
print("Batch LiDAR Example")
print("=" * 60)
print("This example demonstrates batch LiDAR ray tracing.")
print("Add '--profile true' to see performance metrics.")
print("Expected performance: hundred of millions of rays per second (depends on hardware)")
print("=" * 60)
print()

_Lidar = flags.DEFINE_string("lidartype", "mid360", "LiDAR type, Choices: [airy, mid360]")
_VisPoints = flags.DEFINE_string("vispoints", "true", "Visualize points, Choices: [true, false]")
_Profile = flags.DEFINE_string("profile", "false", "Enable profiling, Choices: [true, false]")
_Verbose = flags.DEFINE_string("verbose", "false", "Enable verbose output, Choices: [true, false]")
_TestCount = flags.DEFINE_integer("testcount", 10, "Number of test iterations before exit")
_Stairs = flags.DEFINE_string("stairs", "false", "Use stairs terrain, Choices: [true, false]")
_Backend = flags.DEFINE_string("backend", "jax", "LiDAR backend, Choices: [jax, taichi]")

def main(argv):
    lidar_type = _Lidar.value
    vis_points = _VisPoints.value.lower() == "true"
    test_count = _TestCount.value

    dynamic_lidar = False
    if lidar_type == "airy":
        rays_theta, rays_phi = scan_gen.generate_airy96()
        # downsample for faster rendering
        # rays_theta, rays_phi = rays_theta[::2], rays_phi[::2]
    elif lidar_type == "mid360":
        livox_generator = scan_gen.LivoxGenerator(lidar_type)
        rays_theta, rays_phi = livox_generator.sample_ray_angles(downsample=4)
        dynamic_lidar = True
    rays_theta = np.ascontiguousarray(rays_theta).astype(np.float32)
    rays_phi = np.ascontiguousarray(rays_phi).astype(np.float32)
    num_rays = rays_theta.shape[0]
    cmap = plt.get_cmap('hsv')  # 或使用 'jet', 'viridis', 'plasma' 等

    size = 2
    batch_size = size * size

    if _Stairs.value.lower() == "true":
        path = "examples/assets/go2/scene_stairs_terrain.xml"
    else:
        path = "examples/assets/go2/scene_geom.xml"
    model = load_model(path)
    data = SceneData(model, batch=(batch_size,))
    # a batch dimension is added to all data fields
    assert data.dof_pos.shape == (batch_size, model.num_dof_pos)
    # initialize motrixsim data
    step(model, data)  

    # setup LiDAR wrapper at once
    bridge = MjMxBridge(path)
    bridge.load_keyframe(data, model, "home")
    bridge.update(data)

    # geomgroup = np.ones((6,), dtype=np.ubyte)
    # geomgroup[3:] = 0  # 排除group 1中的几何体
    geomgroup = np.zeros((6,), dtype=np.ubyte)
    geomgroup[2] = 1
    if _Backend.value.lower() == "jax":
        lidar_wrapper = MjLidarJax(
            bridge.mj_model,
            geomgroup=geomgroup,
            bodyexclude=bridge.mj_model.body("base").id
        )
    elif _Backend.value.lower() == "taichi":
        assert mujoco_lidar.__version__ >= "0.2.5", "Please upgrade mujoco-lidar to version 0.2.5 or higher to use Taichi backend for batch rendering."
        import taichi as ti
        ti.init(arch=ti.gpu)

        from mujoco_lidar.core_ti import MjLidarTi
        lidar_wrapper = MjLidarTi(
            bridge.mj_model,
            geomgroup=geomgroup,
            max_candidates=64
        )

        ####################################################################################
        # Important: must call update to sync the mj_data before tracing rays at first time
        lidar_wrapper.update(bridge.mj_data)
        ####################################################################################
    else:
        raise ValueError(f"Unsupported backend: {_Backend.value}. Choose from [jax, taichi].")

    lidar_fps = 12
    lidar_site = model.get_site("lidar")
    geom_xpos_batch_jax = jnp.repeat(jnp.expand_dims(jnp.array(bridge.mj_data.geom_xpos), axis=0), batch_size, axis=0)
    geom_xmat_batch_jax = jnp.repeat(jnp.expand_dims(jnp.array(bridge.mj_data.geom_xmat), axis=0), batch_size, axis=0)

    # distances: (batch_size, num_rays)
    # - 激光雷达射线到碰撞点的距离 (Distance from LiDAR to hit points)
    # local_rays_batch: (batch_size, num_rays, 3)
    # - 激光雷达局部坐标系下的射线单位方向向量 (Unit direction vectors in local frame)
    # local_points: (batch_size, num_rays, 3)
    # - 激光雷达局部坐标系下的碰撞点坐标 (Hit point coordinates in local frame: direction * distance)
    # world_points: (batch_size, num_rays, 3)
    # - 世界坐标系下的碰撞点坐标 (Hit point coordinates in world frame)
    if _Backend.value.lower() == "jax":
        distances, local_rays_batch = lidar_wrapper.trace_rays_batch(
            geom_xpos_batch_jax,
            geom_xmat_batch_jax,
            lidar_site.get_position(data),
            lidar_site.get_rotation_mat(data),
            rays_theta,
            rays_phi
        )
        local_points = local_rays_batch * distances[..., jnp.newaxis]
        world_points = jnp.einsum('bij,bkj->bki', lidar_site.get_rotation_mat(data), local_points) + lidar_site.get_position(data)[:, jnp.newaxis, :]
    elif _Backend.value.lower() == "taichi":
        distances_ti, local_points_ti = lidar_wrapper.trace_rays_batch(
            lidar_site.get_position(data),
            lidar_site.get_rotation_mat(data),
            rays_theta,
            rays_phi
        )
        distances = distances_ti.to_numpy()
        local_points = local_points_ti.to_numpy()
        
        # Calculate world points in numpy
        # local_points is (B, N, 3)
        # sensor_rot is (B, 3, 3)
        # sensor_pos is (B, 3)
        # world = R @ local.T + p
        # But local is (B, N, 3), so we need to transpose local to (B, 3, N) for matmul
        # Then transpose result back to (B, N, 3)
        
        sensor_rot_np = lidar_site.get_rotation_mat(data)
        sensor_pos_np = lidar_site.get_position(data)
        
        # (B, 3, 3) @ (B, 3, N) -> (B, 3, N)
        world_points_T = sensor_rot_np @ local_points.transpose(0, 2, 1)
        world_points = world_points_T.transpose(0, 2, 1) + sensor_pos_np[:, np.newaxis, :]
        
        # For compatibility with JAX path which returns local_rays_batch, we can just mock it or ignore
        local_rays_batch = local_points # Not exactly unit vectors but used for assertions
        
    assert distances.shape == (batch_size, num_rays), f"Expected distances shape {(batch_size, num_rays)}, but got {distances.shape}"
    assert local_rays_batch.shape == (batch_size, num_rays, 3), f"Expected local_rays_batch shape {(batch_size, num_rays, 3)}, but got {local_rays_batch.shape}"
    assert local_points.shape == (batch_size, num_rays, 3), f"Expected local_points shape {(batch_size, num_rays, 3)}, but got {local_points.shape}"
    assert world_points.shape == (batch_size, num_rays, 3), f"Expected world_points shape {(batch_size, num_rays, 3)}, but got {world_points.shape}"
    world_points_np = np.array(world_points)

    # When create scene data in batch mode, we also need to launch the render in batch mode.
    # The render offset can be assigned for each instance to avoid overlapping.
    # Note: The offset only affects the render objects, the physics instance is still at the origin.
    grid_size = (25 * float(size < 3)) if _Stairs.value.lower() == "true" else 25
    render_offset = []
    for i in range(size):
        for j in range(size):
            render_offset.append([-i * grid_size, j * grid_size, 0])
 
    with RenderApp() as render:
        gizmos = render.gizmos
        render.launch(model, batch=batch_size, render_offset=render_offset)

        render_fps = 60
        render_cnt = 0
        lidar_render_substep = int(round(render_fps // lidar_fps))
        
        # Profiling variables
        lidar_test_cnt = 0
        lidar_times = []
        
        def render_step():
            nonlocal dynamic_lidar, livox_generator, lidar_wrapper, rays_theta, rays_phi
            nonlocal geom_xpos_batch_jax, geom_xmat_batch_jax
            nonlocal render_cnt, lidar_render_substep
            nonlocal lidar_site
            nonlocal world_points_np
            nonlocal lidar_test_cnt, lidar_times

            if render_cnt % lidar_render_substep == 0:
                if dynamic_lidar:
                    rays_theta, rays_phi = livox_generator.sample_ray_angles(downsample=4)
                if _Backend.value.lower() == "jax":
                    st = time.time()
                    distances, local_rays_batch = lidar_wrapper.trace_rays_batch(
                        geom_xpos_batch_jax,
                        geom_xmat_batch_jax,
                        lidar_site.get_position(data),
                        lidar_site.get_rotation_mat(data),
                        rays_theta,
                        rays_phi
                    )
                    distances.block_until_ready()
                    et = time.time()
                    local_points = local_rays_batch * distances[..., jnp.newaxis]
                    world_points = jnp.einsum('bij,bkj->bki', lidar_site.get_rotation_mat(data), local_points) + lidar_site.get_position(data)[:, jnp.newaxis, :]
                    world_points_np = np.array(world_points)
                elif _Backend.value.lower() == "taichi":

                    st = time.time()
                    distances_ti, local_points_ti = lidar_wrapper.trace_rays_batch(
                        lidar_site.get_position(data),
                        lidar_site.get_rotation_mat(data),
                        rays_theta,
                        rays_phi
                    )
                    et = time.time()
                    distances = distances_ti.to_numpy()
                    local_points = local_points_ti.to_numpy()

                    sensor_rot_np = lidar_site.get_rotation_mat(data)
                    sensor_pos_np = lidar_site.get_position(data)
                    world_points_T = sensor_rot_np @ local_points.transpose(0, 2, 1)
                    world_points = world_points_T.transpose(0, 2, 1) + sensor_pos_np[:, np.newaxis, :]
                    world_points_np = np.array(world_points)

                lidar_time = et - st
                if lidar_test_cnt:
                    lidar_times.append(lidar_time)
                lidar_test_cnt += 1

                if _Verbose.value.lower() == "true":
                    print(f"[Render] LiDAR batch trace time: {lidar_time*1000:.2f} ms, fps: {1.0/lidar_time:.2f}, batch size: {batch_size}, num rays: {distances.shape[1]}, total rays million/s: {distances.shape[0]*distances.shape[1]/lidar_time/1e6:.2f}")

                if _Profile.value.lower() == "true" and lidar_test_cnt > test_count:
                    # Print statistics
                    times_array = np.array(lidar_times)
                    print("\n" + "=" * 60)
                    print("LiDAR Performance Statistics")
                    print("=" * 60)
                    print(f"Total tests: {len(lidar_times)}")
                    print(f"Batch size: {batch_size}")
                    print(f"Number of rays per batch: {distances.shape[1]}")
                    print(f"Total rays per test: {batch_size * distances.shape[1]}")
                    print(f"Average time: {times_array.mean()*1000:.2f} ms")
                    print(f"Min time: {times_array.min()*1000:.2f} ms")
                    print(f"Max time: {times_array.max()*1000:.2f} ms")
                    print(f"Std deviation: {times_array.std()*1000:.2f} ms")
                    print(f"Average FPS: {1.0/times_array.mean():.2f}")
                    print(f"Average MRays/s: {batch_size * distances.shape[1] / times_array.mean() / 1e6:.2f}")
                    print("=" * 60)
                    
                    import sys
                    sys.exit(0)

            if vis_points:
                # 可视化非常吃性能，训练的时候关掉
                for ib in range(batch_size):
                    # 根据高度设置颜色
                    z_values =  world_points_np[ib, :, 2]
                    z_min, z_max = z_values.min(), z_values.max()
                    if z_max > z_min:
                        # 归一化高度值到 [0, 1]
                        z_norm = (z_values - z_min) / (z_max - z_min)
                    else:
                        z_norm = np.zeros_like(z_values)
                    
                    # 使用 matplotlib 颜色映射
                    colors = cmap(z_norm)  # 返回 RGBA 值，shape: (N, 4)

                    for ir in range(rays_theta.shape[0]):
                        point = world_points_np[ib, ir, :]
                        x, y, z = point + render_offset[ib]
                        r, g, b = colors[ir][:3]
                        gizmos.draw_sphere(0.01, np.array([x, y, z]), color=Color.rgb(r, g, b))

                    # 如果卡的话，可以只渲染一个batch的点云
                    # break

            render.sync(data)
            render_cnt += 1

        run.render_loop(
            model.options.timestep, 
            render_fps, 
            lambda: step(model, data), 
            render_step
        )

if __name__ == "__main__":
    app.run(main)