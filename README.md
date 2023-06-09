# NeRF---
NeRF---: Plus Version of NeRF--, optimize camera poses and NeRF in an end-to-end manner.

### Input & Assumptions
* Images: forward-facing setup
* Intrinsics: all images are captured with same intrinsic parameters
    * Here is different from original NeRF--, which also optimize intrinsic parameters

### Details
* Learnable Parameters
    * Pose: C2W(CameraToWorld) representation![](readme_visuals/c2w_pose.png)
        * we adopt axis-angle representation for pose, as in NeRF--
    * NeRF: vanilla MLP
        * no hierarchy sampling startegy
        * hidden layer dimension is 128 by default
        * kaiming initialization for MLP
        * vanilla positional encoding
            * PE(XYZ): 3 -> 6L+3 (L=10)
            * PE(Direction): 3 -> 6L+3 (L=4)
* Training Implementation
    * 128 points along each ray
    * 2 separate Adam optimizer for pose and NeRF respectively
        * initial learning rate: 0.001
        * NeRF's lr decay per 10 epochs by multiplying 0.9954
        * Pose's lr decay per 100 epochs by multiplying 0.9
    * total 10000 epochs
    * batchsize for inference: 40960 
    * batchsize for train: TODO 

### Support Dataset
 * **Blender**
 * **LLFF** 
    * https://www.robots.ox.ac.uk/~ryan/nerfmm2021/nerfmm_release_data.tar.gz.

### TODO
 * trying to reimplement vanilla NeRF
    * perturb on ray sampling, meaning of this ?
    * visualize the scene and pose, similar to NeRF++, instant-ngp may also provide a good check
    * learning rate decay
    * add LLFF support
 * use eninops
 * HashNeRF
    * xyz bounding_box and such things, device

### STATE
 * June 4. 256 Samples. Using epoch 29. PSNR: 24.966. hidden dim: 128 ![](readme_visuals/hidden128.gif)
 * June 6. 256 Samples. Using epoch 29. PSNR: 27.706. hidden dim: 256 ![](readme_visuals/hidden256.gif)
    * **may be I have not decided a reasonable bound, so PSNR is relative lower than orginal paper(31.42) (near=2, far=6)**
 * June 8. 256 Samples. Using epoch 9. PSNR: 27.809. near = 2, far = 6 ![](readme_visuals/hidden256_near2_far6.gif)
    * **Fluctuation is very high, peak is close to 31, and some outliers are 21**
 * June 10. HashNeRF. PSNR: 30.994 Fix a fatal bug and now everything works great. No fluctuation anymore. ![](readme_visuals/cmp_video.gif) ![](readme_visuals/accmap.gif)

### NOTES on vanilla NeRF
 * Paper description
    * a batch size of 4096 rays
    * each sampled at Nc=64 coordinates in the coarse volume and Nf =128 additional coordinates in the fine volume
    * Adam optimizer with a learning rate that begins at 5 × 10 −4 and decays **exponentially** to 5 × 10 −5 over the course of optimization (other Adam hyperparameters are left at default values of 0.9, 0.999, 10-7).
    * The optimization for a single scene typically take around 100– 300k iterations to converge on a single NVIDIA V100 GPU (**about 1–2 days**)
    ![](readme_visuals/2023-06-06-12-19-53.png)
    ![](readme_visuals/2023-06-06-12-20-21.png)