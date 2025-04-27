# AeroGTO (AAAI 2025)

:triangular_flag_on_post:**News** (2024.12) Our model achieved second place in the CIKM 2024 AnalytiCup competition.

AeroGTO: An Efffcient Graph-Transformer Operator for Learning Large-Scale Aerodynamics of 3D Vehicle Geometries [[Paper]](https://doi.org/10.1609/aaai.v39i18.34083)


In the automotive industry, achieving high-precision aerodynamics requires large-scale computational fluid dynamics (CFD) simulations, which are both time-consuming and computationally expensive. 
To capturing intricate physical correlations across complex geometries while balancing large-scale discretization with computational costs,we propose AeroGTO, an efficient graph-transformer operator designed specifically for large-scale aerodynamics with the following features:

- Local and Global Feature Extraction: **By combining message passing and projection-inspired attention, AeroGTO isolates and captures physical correlations at both local and global levels, enhancing interpretability.**
- Efficient Graph Neural Network: **The frequency-enhanced GNN with kNN handling 3D geometries allows for efficient local feature extraction.**
- Transformer Architecture with Linear Complexity: **The model’s ability to handle multi-level dependencies with linear complexity relative to mesh points allows for fast, scalable inference.**
- Performance Gains: **AeroGTO reduces error by 7.36% on average, achieves a 10.71% improvement in drag coefficient estimation, and provides fast, low-parameter predictions with an overall R² of 0.9250 on unseen data.**

<p align="center">
<img src=".\pic\AeroGTO.png" alt="" align=center />
<br><br>
<b>Figure 1.</b> Overview of AeroGTO.
</p>


## AeroGTO v.s. Previous Operators

Compared to previous state-of-the-art models, our model demonstrates **superior performance** and **reduces computational resource usage significantly** by leveraging attention method.

As shown below, AeroGTO can accurately capture the vehicle's geometric information, enabling precise and efficient surface pressure prediction.

<p align="center">
<img src=".\pic\pressure.png" height = "600" alt="" align=center />
<br><br>
<b>Figure 2.</b> Visualization of ground-truth pressure and corresponding prediction.
</p>

## Get Started

Add dataset folder if it does not exist, add data to corresponding dataset. To make it easier for everyone to run the program, we have included a sample in `ShapeNet dataset` for each data collection. You can directly run the following code to test it in `code` folder:

```python
# For DP training
bash run_dp.sh
# For DDP training
bash run_ddp.sh
```

You should change the environment settings in the file according to your own hardware configuration.



**Data Format:**

The format of our dataset (AeroGTO_Dataset) should be as follows:

```python
Node_pos = [
    [X1, Y1, Z1],
    [X2, Y2, Z2],
   ...
]
Cells = [
    [p1, p2, p3],
    [p4, p5, p6],
   ...
]
```
- **X,Y,Z**: (X_dim x Y_dim x Z_dim) numpy array, representing input mesh points
- X_dim, Y_dim, Z_dim: input dimension of geometry

- **Cells**: (N_cells x 3) numpy array, representing Geometric Cells, which contains three points. The minimum index is `0`, and the maximum value is `N_points-1`

- **Note**: <br />
    I. For a single sample, The number of points must match, i.e, ``X.shape[0]=Y.shape[0]``, but it can vary with different samples. <br />
    II. If additional preprocessing is required for the data, please modify it in the corresponding dataset file in the directory `code/src` <br />

For test, You can directly run the following code to test it in `code` folder to get infer result, saving as `*.npy` in `result` folder:

```python
# For infer on a single GPU
bash infer.sh
```

## Requirements

- torch==2.1.0
- torch_scatter==2.1.0
- numpy==1.24.3
- pandas==2.0.1
- plyfile==0.7.4
- h5py==3.9.0
- vtk==9.2.6
- tensorboardX==2.6


## Citation

If you find this repo useful, please cite our paper. 

```
Liu, P., Wang, P., Ren, X., Yuan, H., Hao, Z., Xu, C., Cai, S., & Ni, D. (2025).
AeroGTO: An Efficient Graph-Transformer Operator for Learning Large-Scale Aerodynamics of 3D Vehicle Geometries.
Proceedings of the AAAI Conference on Artificial Intelligence, 39(18), 18924-18932. 
```

## Contact

If you have any questions or want to use the code, please contact [liupw@zju.edu.cn](mailto:liupw@zju.edu.cn).

## Contributing

We welcome contributions to improve the dataset or project. Please submit pull requests for review.

## Acknowledgement

We appreciate the following contents a lot for their valuable code base or datasets:

https://github.com/echowve/meshGraphNets_pytorch

https://github.com/HaoZhongkai/GNOT

https://github.com/thuml/Transolver

https://github.com/7tl7qns7ch/IPOT

https://github.com/Mohamedelrefaie/DrivAerNet

https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/resources/modulus_datasets-ahmed_body_test
