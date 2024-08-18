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

This repo focuses on the Sequential Multi-Stage Generation approach, and the generation about 3D scene. As for the other topic, please refer to the comprehensive collections listed under [Related-Repos-and-Websites](##Related-Repos-and-Websites) at the end of this file. Feel free to submit a pull request if you have relevant papers to add.

Other repos:

-   **[Text-to-3D](https://paperswithcode.com/task/text-to-3d)** for a carefully compiled collection of text-to-3D research papers.
-   **[Awesome Text-to-3D](https://github.com/yyeboah/Awesome-Text-to-3D)** for a curated list of text-to-3D.

> **About abbreviation:** In the list below: **<span style="color: hsl(20, 100%, 50%);">B</span>** for best paper, **<span style="color: hsl(120, 70%, 50%);">S</span>** for spotlight, **<span style="color: hsl(190, 100%, 50%);">H</span>** for highlight, **<span style="color: hsl(60, 100%, 50%);">W</span>** for workshop.

## History

- **2021.06** - **[Text2Mesh: Text-Driven Neural Stylization for Meshes](https://arxiv.org/abs/2112.03221)** **(CVPR 2022) :** This work develops intuitive controls for editing the style of 3D objects by predicting color and local geometric details based on a target text prompt.
- **2021.12** - **[DreamField: Zero-Shot Text-Guided Object Generation with Dream Fields](https://arxiv.org/abs/2112.01455)** **(CVPR 2022) :** This paper addresses the limitations of DreamFusion by utilizing a two-stage optimization framework to create high-quality 3D mesh models in a shorter time.
- **2022.09** - **[DreamFusion: Text-to-3D using 2D Diffusion](https://arxiv.org/abs/2209.14988)** **(ICLR 2023) :** This paper introduces a method for generating 3D objects using 2D diffusion models.
- **2022.11** - **[Magic3D: High-Resolution Text-to-3D Content Creation](https://arxiv.org/abs/2211.10440)** **(CVPR 2023) :** This paper introduces a method for generating 3D objects using 2D diffusion models. 
- **2023.03** - **[Fantasia3D: Disentangling Geometry and Appearance for High-quality Text-to-3D Content Creation](https://arxiv.org/abs/2303.13873)** **(ICCV 2023) :** This research focuses on disentangling geometry and appearance for high-quality 3D content creation.
- **2023.05** - **[HiFA: High-fidelity Text-to-3D Generation with Advanced Diffusion Guidance](https://arxiv.org/abs/2305.18766)** **(CVPR 2024) :** This paper proposes holistic sampling and smoothing approaches to achieve high-quality text-to-3D generation in a single-stage optimization.
- **2023.10** - **[GaussianDreamer: Fast Generation from Text to 3D Gaussians by Bridging 2D and 3D Diffusion Models](https://arxiv.org/abs/2310.08529)** **(CVPR 2024) :** This paper introduces a novel framework designed to efficiently produce high-quality 3D assets from textual prompts.
- **2023.11** - **[LucidDreamer: Towards High-Fidelity Text-to-3D Generation via Interval Score Matching](https://arxiv.org/abs/2311.11284)** **(CVPR 2024) (<span style="color: hsl(190, 100%, 50%);">H</span>) :** This research introduces a novel method called Interval Score Matching (ISM) for generating high-fidelity 3D models.
- **2023.11** - **[LucidDreamer: Domain-free Generation of 3D Gaussian Splatting Scenes](https://arxiv.org/abs/2311.13384)** **(arXiv 2023) (<span style="color: hsl(60, 150%, 50%);">1.3k stars!</span>) :** This research introduces a novel method called Interval Score Matching (ISM) for generating high-fidelity 3D models.
- **2024.02** - **[GALA3D: Towards Text-to-3D Complex Scene Generation via Layout-guided Generative Gaussian Splatting](https://arxiv.org/abs/2402.07207)** **(ICML 2024) :** This research introduces a novel framework for generating complex 3D scenes from textual descriptions.
- **2024.04** - **[RealmDreamer: Text-Driven 3D Scene Generation with Inpainting and Depth Diffusion](https://arxiv.org/abs/2404.07199)** **(arXiv 2024) :** This research introduces a model to use pretrained inpainting and depth priors with a robust initialization of a 3D Gaussian Splatting model. 
- **2024.06** - **[GradeADreamer: Enhanced Text-to-3D Generation Using Gaussian Splatting and Multi-View Diffusion](https://arxiv.org/abs/2406.09850)** **(arXiv 2024) :** This research introduces a novel three-stage training pipeline called GradeADreamer, which aims to address common challenges in text-to-3D generation, such as the Multi-face Janus problem and extended generation time for high-quality assets.
- **2024.07** - **[PlacidDreamer: Advancing Harmony in Text-to-3D Generation](https://arxiv.org/abs/2407.13976)** **(ACM MM 2024) :** This research explores methods for multi-view consistency and detail optimization.
- **2024.07** - **[ScaleDreamer: Scalable Text-to-3D Synthesis with Asynchronous Score Distillation](https://arxiv.org/abs/2407.02040)** **(ECCV 2024) :** This paper introduces an asynchronous score distillation method to enhance generation quality.
- **2024.08** - **[DreamLCM: Towards High-Quality Text-to-3D Generation via Latent Consistency Model](https://arxiv.org/abs/2408.02993)** **(ACM MM 2024) :** This paper proposes a method to improve 3D generation quality through a latent consistency model.

## Papers

### **2020**
- **2020.02** - **[CDISN: Deep Implicit Surface Network for High-quality Single-view 3D Reconstruction](https://arxiv.org/abs/1905.10711)** **(CVPR 2020)**
- **2020.03** - **[3D Photography using Context-aware Layered Depth Inpainting](https://arxiv.org/abs/2004.04727)** **(CVPR 2020)** 
- **2020.03** - **[Learning Implicit Fields for Generative Shape Modeling](https://arxiv.org/abs/1905.10711)** **(CVPR 2020)**
- **2020.03** - **[Pix2Vox++: Multi-Scale Context-Aware 3D Object Reconstruction from Single and Multiple Images](https://arxiv.org/abs/1901.11153)** **(CVPR 2020)**
- **2020.03** - **[Neural Mesh Flow: 3D Manifold Mesh Generation via Diffeomorphic Flows](https://arxiv.org/abs/2007.10973)** **(CVPR 2020)**
-  **2020.03** - **[Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes](https://arxiv.org/abs/2011.13084)** **(CVPR 202)**

### **2021**
- **2021.03** - **[Text2Shape: Generating Shapes from Natural Language by Learning Joint Embedding](https://arxiv.org/abs/1803.08495)** **(CVPR 2021)**
- **2021.06** - **[Text2Mesh: Text-Driven Neural Stylization for Meshes](https://arxiv.org/abs/2112.03221)** **(SIGGRAPH 2021)**
- **2021.09** - **[CLIP-Forge: Towards Zero-Shot Text-to-Shape Generation](https://arxiv.org/abs/2110.02624)** **(NeurIPS 2021)**

### **2022**
- **2022.09** - **[DreamFusion: Text-to-3D using 2D Diffusion](https://arxiv.org/abs/2209.14988)** **(ICRL 2023)**
- **2022.11** - **[Magic3D: High-Resolution Text-to-3D Content Creation](https://arxiv.org/abs/2211.10440)** **(CVPR 2023)**

### **2023**
- **2023.03** - **[Fantasia3D: Disentangling Geometry and Appearance for High-quality Text-to-3D Content Creation](https://arxiv.org/abs/2303.13873)** **(ICCV 2023)**
- **2023.05** - **[HiFA: High-fidelity Text-to-3D Generation with Advanced Diffusion Guidance](https://arxiv.org/abs/2305.18766)** **(CVPR 2024)** 
- **2023.08** - **[IT3D: Improved Text-to-3D Generation with Explicit View Synthesis](https://arxiv.org/abs/2308.11473)** **(AAAI 2024)**
- **2023.10** - **[GaussianDreamer: Fast Generation from Text to 3D Gaussians by Bridging 2D and 3D Diffusion Models](https://arxiv.org/abs/2310.08529)** **(CVPR 2024)**
- **2023.11** - **[LucidDreamer: Towards High-Fidelity Text-to-3D Generation via Interval Score Matching](https://arxiv.org/abs/2311.11284)** **(CVPR 2024) (<span style="color: hsl(190, 100%, 50%);">H</span>) :** This research introduces a novel method called Interval Score Matching (ISM) for generating high-fidelity 3D models.
- **2023.11** - **[LucidDreamer: Domain-free Generation of 3D Gaussian Splatting Scenes](https://arxiv.org/abs/2311.13384)** **(arXiv 2023) (<span style="color: hsl(60, 150%, 50%);">1.3k stars!</span>) :** This research introduces a novel method called Interval Score Matching (ISM) for generating high-fidelity 3D models.
- **2023.12** - **[Sherpa3D: Boosting High-Fidelity Text-to-3D Generation via Coarse 3D Prior](https://arxiv.org/abs/2312.06655)** **(CVPR 2024)** 

### **2024**
- **2024.02** - **[GALA3D: Towards Text-to-3D Complex Scene Generation via Layout-guided Generative Gaussian Splatting](https://arxiv.org/abs/2402.07207)** **(ICML 2024)** 
- **2024.04** - **[RealmDreamer: Text-Driven 3D Scene Generation with Inpainting and Depth Diffusion](https://arxiv.org/abs/2404.07199)** **(arXiv 2024)**
- **2024.06** - **[GradeADreamer: Enhanced Text-to-3D Generation Using Gaussian Splatting and Multi-View Diffusion](https://arxiv.org/abs/2406.09850)** **（arXiv 2024)**
- **2024.07** - **[PlacidDreamer: Advancing Harmony in Text-to-3D Generation](https://arxiv.org/abs/2407.13976)** **(ACM MM 2024)** 
- **2024.07** - **[ScaleDreamer: Scalable Text-to-3D Synthesis with Asynchronous Score Distillation](https://arxiv.org/abs/2407.02040)** **(ECCV 2024)** 
- **2024.08** - **[DreamLCM: Towards High-Quality Text-to-3D Generation via Latent Consistency Model](https://arxiv.org/abs/2408.02993)** **(ACM MM 2024)** 

## Related Repos and Websites
- **[Awesome Text-to-3D](https://github.com/yyeboah/Awesome-Text-to-3D)**
- **[Text-to-3D](https://paperswithcode.com/task/text-to-3d)**

