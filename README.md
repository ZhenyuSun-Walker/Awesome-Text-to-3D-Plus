# Text to 3D Scene Generation
A collection of recent methods on 3D scene generation from text description.
There are mainly 2 kinds of methods of text to 3D generation:

- Direct End-to-End Generation
(There are multiple internal steps, but they are transparent to the user)
    - initialize a coarse layout from text, and then refine/inpaint it
    - generate a local scene from text, and then outpaint/optimize it 
- Sequential Multi-Stage Generation
(Each internal step has an independent output as the input for the next stage)
    - reconstruction based on text-to-image models and depth-estimation models
    - reconstruct based on the multi-view generation models from text
    - reconstruct a premitive scene from a text-to-image model, then gradually expand it and align features

This repo focuses on the Sequential Multi-Stage Generation approach. As for the other topic, please refer to the comprehensive collections listed under ‘Related Repos and Websites’ at the end of this file. Feel free to submit a pull request if you have relevant papers to add.

Other repos:

-   LTH (lottery ticket hypothesis) and its broader version, _pruning at initialization (PaI)_, now is at the frontier of network pruning. We single out the PaI papers to [this repo](https://github.com/MingSun-Tse/Awesome-Pruning-at-Initialization). Welcome to check it out!
-   [Awesome-Efficient-ViT](https://github.com/MingSun-Tse/Awesome-Efficient-ViT) for a curated list of efficient vision transformers.

> About abbreviation: In the list below, `o` for oral, `s` for spotlight, `b` for best paper, `w` for workshop.