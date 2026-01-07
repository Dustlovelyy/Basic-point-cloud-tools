# Basic-point-cloud-tools
Just some tools might used in Point Cloud Completion... for personal sharing of some utils during my study.

If it works on my PC, then it works.
~~Not guaranteed to be 100% correct.~~
#
I will introduce this section using language that is as concise, comprehensive, and intuitive as possible, regardless of whether you have prior experience with point clouds or other 3D spatial data.
## Farthest Point Sampling (FPS)
***FPS*** stands for Farthest Point Sampling, and it is one of the most commonly used ***downsampling*** methods for point clouds.

The basic idea is very simple. First, you start by choosing a point A from the point set. This point can be selected randomly, or you can fix it in advance — both choices are acceptable. Then, you compute the distances between point A and all the remaining points in the point cloud, and select the point B that has the largest distance to point A.

At this point, you already have two representative points (A and B) for the whole point set. Next, you repeat the same process: for every point that has not been selected yet, you compute its distance to the current sampled points, and then choose the one that is farthest from the existing sampled set.

You keep looping this procedure until the number of sampled points reaches your desired count. In this way, FPS gradually selects points that are well spread out over the entire point cloud, instead of clustering in local regions.

you can found standard FPS by: [fpsample](https://github.com/leonardodalinky/fpsample)
### this part refers to file:"fps.py"
I define three functions in this file.

farthest_point_sample and farthest_point_sample_batch are standard FPS implementations.

farthest_voxel_points_samplings is a ***voxel-based*** FPS, which adjusts the downsampling rate in different voxels and usually shows better results when dealing with ***non-uniform*** point cloud data, but it is much ***slower*** because it relies on Open3D, which only runs on the CPU.
