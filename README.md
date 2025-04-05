# PMGS: Reconstruction of Projectile Motion with 3D Gaussian Splatting
![teaser](https://github.com/user-attachments/assets/38ead2a6-f638-4023-84cd-f636b71a0909)

## Abstract
Projectile motion reconstruction is critical for robotics and computer vision applications. However, jointly estimating 3D geometry and motion parameters from monocular observations remains challenging due to its ill-posed inverse nature, while existing dynamic neural rendering methods struggle to capture complex rigid-body physics across large spatial-temporal scales. To address these issues, we propose a unified framework that integrates static modeling with dynamic tracking for projectile motion reconstruction. Our approach comprises two key stages: 1) Dynamic scene decomposition for appearance and geometry modeling, where the complex motion is converted into equivalent static scenes via a Focus-Align module, reconstructed through optical flow-enhanced Gaussian splatting with an improved point density control mechanism; 2) Physically-constrained motion tracking for trajectory restoration, where we estimate per-frame SE(3) transformations based on explicit Gaussian representations, and enforce Newtonian acceleration priors to ensure physically consistent motion. To further accommodate time-varying motion states, we introduce a Dynamic Simulated Annealing (DSA) strategy that adaptively schedules training processes, effectively eliminating oscillations and trajectory fractures caused by conventional fixed training paradigms. Experiments on both synthetic and real-world datasets demonstrate that our method achieves efficient target reconstruction while recovering complete projectile motion trajectories. 

## Methods
![Methods](https://github.com/user-attachments/assets/2099ac39-543b-499e-9a4f-19295ead4eec)
Overview of PMGS. For modeling appearance and geometry, we first segment the target via the pre-trained SAM model, then decompose the motion through centralization to transform the dynamic scene into static. Following the 3DGS pipeline, we reconstruct a set of Gaussian kernels and align them at the original scale with a set of learnable affine transformation. In the motion restoration stage, we learn the target's 6DoF spatial transformation frame by frame based on the explicit Gaussian representation, and comprehensively improve tracking accuracy by integrating physics-enhanced strategies.

## Demo
### (1) Synthetic（Left-Render; Right-GT）：

<p align="left">
  <img src="https://github.com/user-attachments/assets/74642537-f5a6-4394-aaac-134738a151b5" width="48%">
  <img src="https://github.com/user-attachments/assets/20bb5697-3745-4f13-97f9-be9e80bb7831" width="48%">
</p>

<p align="left">
  <img src="https://github.com/user-attachments/assets/bcd0f7ce-c3d2-4d6a-9001-9d986e2ee707" width="48%"> 
  <img src="https://github.com/user-attachments/assets/fdf52a20-5dd3-459d-beff-4009b1199c92" width="48%">
</p>

<p align="left">
  <img src="https://github.com/user-attachments/assets/b288ca64-47eb-4b19-bad1-1967ee51251f" width="48%">
</p>

### (2) Real：
![box_full_results-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/3f199596-4f85-47db-b5da-7a57d7a98432)

![bear_results-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/a5348f9b-964e-4e89-9449-51ac65158574)

![sb_full_results-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/3c615cdf-8b1a-496e-ac3f-59bc8d483377)
