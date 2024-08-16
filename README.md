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
