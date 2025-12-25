# Copyright (C) 2020-2025 Motphys Technology Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

import mujoco
from motrixsim import load_model, run
from motrixsim.render import Color, RenderApp

from absl import app, flags

from go2_keyboard_control import (
    OnnxController,
    default_joint_pos,
    action_scale,
    lin_vel_scale,
    ang_vel_scale
)
from mujoco.mjmx_bridge import MjMxBridge

try:
    from mujoco_lidar import scan_gen
    from mujoco_lidar import MjLidarWrapper
except ImportError:
    print("[ERROR] mujoco_lidar package not found. Please install mujoco-lidar to run this example.")
    print("Visit https://github.com/TATP-233/MuJoCo-LiDAR for installation instructions.")
    exit(0)

_Lidar = flags.DEFINE_string("lidartype", "mid360", "LiDAR type, Choices: [airy, mid360]")

def main(argv):
    # Prepare the LiDAR scanner configuration
    lidar_fps = 12
    lidar_type = _Lidar.value
    if lidar_type == "airy":
        rays_theta, rays_phi = scan_gen.generate_airy96()
    elif lidar_type == "mid360":
        livox_generator = scan_gen.LivoxGenerator(lidar_type)
        rays_theta, rays_phi = livox_generator.sample_ray_angles()
        dynamic_lidar = True
    rays_theta, rays_phi = livox_generator.sample_ray_angles()
    rays_theta = np.ascontiguousarray(rays_theta).astype(np.float32)
    rays_phi = np.ascontiguousarray(rays_phi).astype(np.float32)
    
    cmap = plt.get_cmap('hsv')  # 或使用 'jet', 'viridis', 'plasma' 等

    # Create render window for visualization
    with RenderApp() as render:
        gizmos = render.gizmos

        # The scene description file
        path = "examples/assets/go2/scene_geom.xml"
        # Load the scene model
        model = load_model(path)

        # Create the render instance of the model
        render.launch(model)

        policy = OnnxController(
            model,
            policy_path="examples/assets/go2/go2_policy.onnx",
            ctrl_dt=0.02,
            default_angles=default_joint_pos,
            action_scale=action_scale,
        )

        bridge = MjMxBridge(path)
        bridge.load_keyframe(policy.data, model, "home")

        geomgroup = np.ones((mujoco.mjNGROUP,), dtype=np.ubyte)
        geomgroup[3:] = 0  # 排除group 1中的几何体
        lidar = MjLidarWrapper(bridge._mj_model, site_name="lidar", backend="taichi", args={'bodyexclude': bridge._mj_model.body("base").id, "geomgroup":geomgroup})
        world_points, colors = None, None

        render_cnt = 0
        lidar_render_substep = int(round(60 // lidar_fps))
        input = render.input
        def render_step():
            nonlocal dynamic_lidar, livox_generator, lidar
            nonlocal render_cnt, lidar_render_substep
            nonlocal world_points, colors
            if render_cnt % lidar_render_substep == 0:
                mj_data = bridge.update(policy.data)
                if dynamic_lidar:
                    rays_theta, rays_phi = livox_generator.sample_ray_angles()
                lidar.trace_rays(mj_data, rays_theta, rays_phi)
                points = lidar.get_hit_points()
                world_points = points @ lidar.sensor_rotation.T + lidar.sensor_position

                # 根据高度设置颜色
                z_values = world_points[:, 2]
                z_min, z_max = z_values.min(), z_values.max()
                if z_max > z_min:
                    # 归一化高度值到 [0, 1]
                    # z_norm = (z_values - z_min) / (z_max - z_min)
                    z_norm = (z_max - z_values) / (z_max - z_min)
                else:
                    z_norm = np.zeros_like(z_values)
                
                # 使用 matplotlib 颜色映射
                colors = cmap(z_norm)  # 返回 RGBA 值，shape: (N, 4)
            
            if world_points is not None and colors is not None:
                for i in range(world_points.shape[0])[::2]:
                    x, y, z = world_points[i]
                    r, g, b, _ = colors[i]
                    gizmos.draw_sphere(0.01, np.array([x, y, z]), color=Color.rgb(r, g, b))

            if input.is_key_pressed("up") or input.is_key_pressed("w"):
                policy.command[0] = 1. * lin_vel_scale
            elif input.is_key_pressed("down") or input.is_key_pressed("s"):
                policy.command[0] = -1. * lin_vel_scale
            else:
                policy.command[0] = 0.

            if input.is_key_pressed("left"):
                policy.command[1] = 0.5 * lin_vel_scale
            elif input.is_key_pressed("right"):
                policy.command[1] = -0.5 * lin_vel_scale
            else:
                policy.command[1] = 0.

            if input.is_key_pressed("a"):
                policy.command[2] = 2. * ang_vel_scale
            elif input.is_key_pressed("d"):
                policy.command[2] = -2. * ang_vel_scale
            else:
                policy.command[2] = 0.

            render.sync(policy.data)
            render_cnt += 1
        
        print("Keyboard Controls:")
        print("- Press W / Up Arrow to move forward")
        print("- Press S / Down Arrow to move backward")
        print("- Press Left Arrow to move left")
        print("- Press Right Arrow to move right")
        print("- Press A to rotate left")
        print("- Press D to rotate right")

        run.render_loop(model.options.timestep, 60, policy.get_control, render_step)
# endtag

if __name__ == "__main__":
    app.run(main)
