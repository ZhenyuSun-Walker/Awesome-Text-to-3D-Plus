# Text-to-3D Generation
A collection of recent methods on 3D generation from text description.
There are mainly 2 kinds of methods of text-to-3D generation:

- **Direct End-to-End Generation**
(There are multiple internal steps, but they are transparent to the user)
    - initialize a coarse layout from text, and then refine/inpaint it
    - generate a local scene from text, and then outpaint/optimize it 
- **Sequential Multi-Stage Generation**
(Each internal step has an independent output as the input for the next stage)
    - reconstruction based on text-to-image models and depth-estimation models
    - reconstruct based on the multi-view generation models from text
    - reconstruct a premitive scene from a text-to-image model, then gradually expand it and align features

This repo focuses on the Sequential Multi-Stage Generation approach, and the generation about 3D scene. As for the other topic, please refer to the comprehensive collections listed under ‘Related Repos and Websites’ at the end of this file. Feel free to submit a pull request if you have relevant papers to add.

Other repos:

-   **[Text-to-3D](https://paperswithcode.com/task/text-to-3d)** for a carefully compiled collection of text-to-3D research papers.
-   **[Awesome Text-to-3D](https://github.com/yyeboah/Awesome-Text-to-3D)** for a curated list of text-to-3D.

> **About abbreviation:** In the list below: **<span style="color: hsl(20, 100%, 50%);">o</span>** for oral, **<span style="color: hsl(120, 70%, 50%);">s</span>** for spotlight, **<span style="color: hsl(190, 100%, 50%);">b</span>** for best paper, **<span style="color: hsl(60, 100%, 50%);">w</span>** for workshop.

## History

- **2021.06** - **[Text2Mesh: Text-Driven Neural Stylization for Meshes](https://arxiv.org/abs/2112.03221)** **(CVPR 2022) :** This work develops intuitive controls for editing the style of 3D objects by predicting color and local geometric details based on a target text prompt.
- **2021.12** - **[DreamField: Zero-Shot Text-Guided Object Generation with Dream Fields](https://arxiv.org/pdf/2112.01455)** **(CVPR 2022) :** This paper addresses the limitations of DreamFusion by utilizing a two-stage optimization framework to create high-quality 3D mesh models in a shorter time.
- **2022.09** - **[DreamFusion: Text-to-3D using 2D Diffusion](https://arxiv.org/pdf/2209.14988)** **(ICLR 2022) :** This paper introduces a method for generating 3D objects using 2D diffusion models.
- **2022.11** - **[Magic3D: High-Resolution Text-to-3D Content Creation](https://arxiv.org/abs/2211.10440)** **(CVPR 2023) :** This paper introduces a method for generating 3D objects using 2D diffusion models. 
- **2023.03** - **[Fantasia3D: Disentangling Geometry and Appearance for High-quality Text-to-3D Content Creation](https://arxiv.org/pdf/2303.13873)** **(ICCV 2023) :** This research focuses on disentangling geometry and appearance for high-quality 3D content creation.
- **2023.05** - **[HiFA: High-fidelity Text-to-3D Generation with Advanced Diffusion Guidance](https://arxiv.org/abs/2305.18766)** **(CVPR 2024) :** This paper proposes holistic sampling and smoothing approaches to achieve high-quality text-to-3D generation in a single-stage optimization.
- **2023.10** - **[GaussianDreamer: Fast Generation from Text to 3D Gaussians by Bridging 2D and 3D Diffusion Models](https://arxiv.org/pdf/2310.08529)** **(CVPR 2024) :** This paper introduces a novel framework designed to efficiently produce high-quality 3D assets from textual prompts.
- **2024.02** - **[GALA3D: Towards Text-to-3D Complex Scene Generation via Layout-guided Generative Gaussian Splatting](https://arxiv.org/abs/2402.07207)** **(ICML 24) :** This research introduces a novel framework for generating complex 3D scenes from textual descriptions.
- **2024.06** - **[GradeADreamer: Enhanced Text-to-3D Generation Using Gaussian Splatting and Multi-View Diffusion](https://arxiv.org/pdf/2406.09850)** **(CVPR 2024) :** This research introduces a novel three-stage training pipeline called GradeADreamer, which aims to address common challenges in text-to-3D generation, such as the Multi-face Janus problem and extended generation time for high-quality assets.
- **2024.07** - **[PlacidDreamer: Advancing Harmony in Text-to-3D Generation](https://arxiv.org/pdf/2407.13976)** **(ACM MM'24) :** This research explores methods for multi-view consistency and detail optimization.
- **2024.07** - **[ScaleDreamer: Scalable Text-to-3D Synthesis with Asynchronous Score Distillation](https://arxiv.org/pdf/2407.02040)** **(ECCV 24) :** This paper introduces an asynchronous score distillation method to enhance generation quality.
- **2024.08** - **[DreamLCM: Towards High-Quality Text-to-3D Generation via Latent Consistency Model](https://arxiv.org/pdf/2408.02993)** **(ACM MM '24) :** This paper proposes a method to improve 3D generation quality through a latent consistency model.

## Papers

**2020s**
-   2017-ICLR-[Pruning Filters for Efficient ConvNets](https://openreview.net/forum?id=rJqFGTslg) [[PyTorch Reimpl. #1](https://github.com/Eric-mingjie/rethinking-network-pruning/tree/master/imagenet/l1-norm-pruning)] [[PyTorch Reimpl. #2](https://github.com/MingSun-Tse/Regularization-Pruning)]
-   2017-ICLR-[Pruning Convolutional Neural Networks for Resource Efficient Inference](https://openreview.net/forum?id=SJGCiw5gl&noteId=SJGCiw5gl)
-   2017-ICLR-[Incremental Network Quantization: Towards Lossless CNNs with Low-Precision Weights](https://arxiv.org/abs/1702.03044) [[Code](https://github.com/Mxbonn/INQ-pytorch)]
-   2017-ICLR-[Do Deep Convolutional Nets Really Need to be Deep and Convolutional?](https://arxiv.org/abs/1603.05691)
-   2017-ICLR-[DSD: Dense-Sparse-Dense Training for Deep Neural Networks](https://arxiv.org/abs/1607.04381)
-   2017-ICLR-[Faster CNNs with Direct Sparse Convolutions and Guided Pruning](https://arxiv.org/abs/1608.01409)
-   2017-ICLR-[Towards the Limit of Network Quantization](https://openreview.net/forum?id=rJ8uNptgl)
-   2017-ICLR-[Loss-aware Binarization of Deep Networks](https://openreview.net/forum?id=S1oWlN9ll&noteId=S1oWlN9ll)
-   2017-ICLR-[Trained Ternary Quantization](https://openreview.net/forum?id=S1_pAu9xl&noteId=S1_pAu9xl) [[Code](https://github.com/czhu95/ternarynet)]
-   2017-ICLR-[Exploring Sparsity in Recurrent Neural Networks](https://openreview.net/forum?id=BylSPv9gx&noteId=BylSPv9gx)
-   2017-ICLR-[Soft Weight-Sharing for Neural Network Compression](https://openreview.net/forum?id=HJGwcKclx) [[Reddit discussion](https://www.reddit.com/r/MachineLearning/comments/5u7h3l/r_compressing_nn_with_shannons_blessing/)] [[Code](https://github.com/KarenUllrich/Tutorial-SoftWeightSharingForNNCompression)]
-   2017-ICLR-[Variable Computation in Recurrent Neural Networks](https://openreview.net/forum?id=S1LVSrcge&noteId=S1LVSrcge)
-   2017-ICLR-[Training Compressed Fully-Connected Networks with a Density-Diversity Penalty](https://openreview.net/forum?id=Hku9NK5lx)
-   2017-ICML-[Theoretical Properties for Neural Networks with Weight Matrices of Low Displacement Rank](https://arxiv.org/abs/1703.00144)
-   2017-ICML-[Deep Tensor Convolution on Multicores](http://proceedings.mlr.press/v70/budden17a.html)
-   2017-ICML-[Delta Networks for Optimized Recurrent Network Computation](http://proceedings.mlr.press/v70/neil17a.html)
-   2017-ICML-[Beyond Filters: Compact Feature Map for Portable Deep Model](http://proceedings.mlr.press/v70/wang17m.html)
-   2017-ICML-[Combined Group and Exclusive Sparsity for Deep Neural Networks](http://proceedings.mlr.press/v70/yoon17a.html)
-   2017-ICML-[MEC: Memory-efficient Convolution for Deep Neural Network](http://proceedings.mlr.press/v70/cho17a.html)
-   2017-ICML-[Deciding How to Decide: Dynamic Routing in Artificial Neural Networks](http://proceedings.mlr.press/v70/mcgill17a.html)
-   2017-ICML-[ZipML: Training Models with End-to-End Low Precision: The Cans, the Cannots, and a Little Bit of Deep Learning](http://proceedings.mlr.press/v70/zhang17e.html)
-   2017-ICML-[Analytical Guarantees on Numerical Precision of Deep Neural Networks](http://proceedings.mlr.press/v70/sakr17a.html)
-   2017-ICML-[Adaptive Neural Networks for Efficient Inference](http://proceedings.mlr.press/v70/bolukbasi17a.html)
-   2017-ICML-[SplitNet: Learning to Semantically Split Deep Networks for Parameter Reduction and Model Parallelization](http://proceedings.mlr.press/v70/kim17b.html)
-   2017-CVPR-[Learning deep CNN denoiser prior for image restoration](http://openaccess.thecvf.com/content_cvpr_2017/html/Zhang_Learning_Deep_CNN_CVPR_2017_paper.html)
-   2017-CVPR-[Deep roots: Improving cnn efficiency with hierarchical filter groups](http://openaccess.thecvf.com/content_cvpr_2017/html/Ioannou_Deep_Roots_Improving_CVPR_2017_paper.html)
-   2017-CVPR-[More is less: A more complicated network with less inference complexity](http://openaccess.thecvf.com/content_cvpr_2017/html/Dong_More_Is_Less_CVPR_2017_paper.html) [[PyTorch Code](https://github.com/D-X-Y/DXY-Projects/tree/master/LCCL)]
-   2017-CVPR-[All You Need is Beyond a Good Init: Exploring Better Solution for Training Extremely Deep Convolutional Neural Networks with Orthonormality and Modulation](http://openaccess.thecvf.com/content_cvpr_2017/html/Xie_All_You_Need_CVPR_2017_paper.html)
-   2017-CVPR-ResNeXt-[Aggregated Residual Transformations for Deep Neural Networks](http://openaccess.thecvf.com/content_cvpr_2017/html/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.html)
-   2017-CVPR-[Xception: Deep learning with depthwise separable convolutions](http://openaccess.thecvf.com/content_cvpr_2017/html/Chollet_Xception_Deep_Learning_CVPR_2017_paper.html)
-   2017-CVPR-[Designing Energy-Efficient CNN using Energy-aware Pruning](http://openaccess.thecvf.com/content_cvpr_2017/html/Yang_Designing_Energy-Efficient_Convolutional_CVPR_2017_paper.html)
-   2017-CVPR-[Spatially Adaptive Computation Time for Residual Networks](http://openaccess.thecvf.com/content_cvpr_2017/html/Figurnov_Spatially_Adaptive_Computation_CVPR_2017_paper.html)
-   2017-CVPR-[Network Sketching: Exploiting Binary Structure in Deep CNNs](http://openaccess.thecvf.com/content_cvpr_2017/html/Guo_Network_Sketching_Exploiting_CVPR_2017_paper.html)
-   2017-CVPR-[A Compact DNN: Approaching GoogLeNet-Level Accuracy of Classification and Domain Adaptation](http://openaccess.thecvf.com/content_cvpr_2017/html/Wu_A_Compact_DNN_CVPR_2017_paper.html)
-   2017-ICCV-[Channel pruning for accelerating very deep neural networks](http://openaccess.thecvf.com/content_iccv_2017/html/He_Channel_Pruning_for_ICCV_2017_paper.html) [[Caffe Code](https://github.com/yihui-he/channel-pruning)]
-   2017-ICCV-[Learning efficient convolutional networks through network slimming](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html) [[PyTorch Code](https://github.com/liuzhuang13/slimming/)]
-   2017-ICCV-[ThiNet: A filter level pruning method for deep neural network compression](http://openaccess.thecvf.com/content_iccv_2017/html/Luo_ThiNet_A_Filter_ICCV_2017_paper.html) [[Project](http://lamda.nju.edu.cn/luojh/project/ThiNet_ICCV17/ThiNet_ICCV17.html)] [[Caffe Code](https://github.com/Roll920/ThiNet_Code)] [[2018 TPAMI version](https://ieeexplore.ieee.org/document/8416559)]
-   2017-ICCV-[Interleaved group convolutions](http://openaccess.thecvf.com/content_iccv_2017/html/Zhang_Interleaved_Group_Convolutions_ICCV_2017_paper.html)
-   2017-ICCV-[Coordinating Filters for Faster Deep Neural Networks](http://openaccess.thecvf.com/content_iccv_2017/html/Wen_Coordinating_Filters_for_ICCV_2017_paper.html) [[Caffe Code](https://github.com/wenwei202/caffe)]
-   2017-ICCV-[Performance Guaranteed Network Acceleration via High-Order Residual Quantization](http://openaccess.thecvf.com/content_iccv_2017/html/Li_Performance_Guaranteed_Network_ICCV_2017_paper.html)
-   2017-NIPS-[Net-trim: Convex pruning of deep neural networks with performance guarantee](http://papers.nips.cc/paper/6910-net-trim-convex-pruning-of-deep-neural-networks-with-performance-guarantee) [[Code](https://github.com/DNNToolBox/Net-Trim)] (Journal version: [2020-SIAM-Fast Convex Pruning of Deep Neural Networks](https://epubs.siam.org/doi/abs/10.1137/19M1246468))
-   2017-NIPS-[Runtime neural pruning](http://papers.nips.cc/paper/6813-runtime-neural-pruning)
-   2017-NIPS-[Learning to Prune Deep Neural Networks via Layer-wise Optimal Brain Surgeon](http://papers.nips.cc/paper/7071-learning-to-prune-deep-neural-networks-via-layer-wise-optimal-brain-surgeon) [[Code](https://github.com/csyhhu/L-OBS)]
-   2017-NIPS-[Federated Multi-Task Learning](http://papers.nips.cc/paper/7029-federated-multi-task-learning)
-   2017-NIPS-[Towards Accurate Binary Convolutional Neural Network](http://papers.nips.cc/paper/6638-towards-accurate-binary-convolutional-neural-network)
-   2017-NIPS-[Soft-to-Hard Vector Quantization for End-to-End Learning Compressible Representations](http://papers.nips.cc/paper/6714-soft-to-hard-vector-quantization-for-end-to-end-learning-compressible-representations)
-   2017-NIPS-[TernGrad: Ternary Gradients to Reduce Communication in Distributed Deep Learning](http://papers.nips.cc/paper/6749-terngrad-ternary-gradients-to-reduce-communication-in-distributed-deep-learning)
-   2017-NIPS-[Flexpoint: An Adaptive Numerical Format for Efficient Training of Deep Neural Networks](http://papers.nips.cc/paper/6771-flexpoint-an-adaptive-numerical-format-for-efficient-training-of-deep-neural-networks)
-   2017-NIPS-[Training Quantized Nets: A Deeper Understanding](http://papers.nips.cc/paper/7163-training-quantized-nets-a-deeper-understanding)
-   2017-NIPS-[The Reversible Residual Network: Backpropagation Without Storing Activations](http://papers.nips.cc/paper/6816-the-reversible-residual-network-backpropagation-without-storing-activations) [[Code](https://github.com/renmengye/revnet-public)]
-   2017-NIPS-[Compression-aware Training of Deep Networks](http://papers.nips.cc/paper/6687-compression-aware-training-of-deep-networks)
-   2017-FPGA-[ESE: efficient speech recognition engine with compressed LSTM on FPGA](https://pdfs.semanticscholar.org/99d2/07c18ba48e41560f3081ea1b7c6fde98c1ce.pdf) [Best paper!]
-   2017-AISTATS-[Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)
-   2017-ICASSP-[Accelerating Deep Convolutional Networks using low-precision and sparsity](https://arxiv.org/abs/1610.00324)
-   2017-NNs-[Nonredundant sparse feature extraction using autoencoders with receptive fields clustering](https://www.sciencedirect.com/science/article/pii/S0893608017300928)
-   2017.02-[The Power of Sparsity in Convolutional Neural Networks](https://arxiv.org/abs/1702.06257)
-   2017.07-[Stochastic, Distributed and Federated Optimization for Machine Learning](https://arxiv.org/abs/1707.01155)
-   2017.05-[Structural Compression of Convolutional Neural Networks Based on Greedy Filter Pruning](https://arxiv.org/abs/1705.07356)
-   2017.07-[Extremely Low Bit Neural Network: Squeeze the Last Bit Out with ADMM](https://arxiv.org/abs/1707.09870)
-   2017.11-[GPU Kernels for Block-Sparse Weights](https://openai.com/blog/block-sparse-gpu-kernels/) [[Code](https://github.com/openai/blocksparse)] (OpenAI)
-   2017.11-[Block-sparse recurrent neural networks](https://arxiv.org/abs/1711.02782)