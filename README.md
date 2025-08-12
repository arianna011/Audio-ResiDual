# Audio-ResiDual
A project about exploring the effectiveness of applying the ResiDual spectral alignment method to the audio domain, working with the CLAP model. Developed for the Deep Learning course at Sapienza a.y. 2024/2025

The code is organized as follows:

1. the main experiments are contained in the 3 sequential notebooks:
    - [1_Analyze_CLAP_Attention.ipynb](https://github.com/arianna011/Audio-ResiDual/blob/main/1_Analyze_CLAP_Attention.ipynb) includes an illustration of CLAP architecture and the formalization needed to properly extract attention head representantions, which are then  analyzed with PCA in order to study the intrinsic dimensionality and specialization of the corresponding spanned latent manifolds;
    - [2_Apply_ResiDual_to_CLAP](https://github.com/arianna011/Audio-ResiDual/blob/main/2_Apply_ResiDual_to_CLAP.ipynb) contains the code to compute the PCA values needed by ResiDual modules and to run the W&B sweep to tune the hyperparameters for ResiDual training;
    - [3_Evaluate_and_compare_CLAP_performance](https://github.com/arianna011/Audio-ResiDual/blob/main/3_Evaluate_and_compare_CLAP_performance.ipynb) includes the training and final comparative evaluation of the version of CLAP featuring injected ResiDual modules with  a version of CLAP leveraging a standard linear classifier for audio classification, with the pretrained CLAP checkpoint as a baseline.
  
2. the notebooks import code from:
    -  [CLAP](https://github.com/arianna011/Audio-ResiDual/tree/main/CLAP): an extension of the [original repository](https://github.com/LAION-AI/CLAP) for the CLAP model that supports attention representations extraction;
    -  [data_processing](https://github.com/arianna011/Audio-ResiDual/tree/main/data_processing): a module for downloading datasets and generating dataloaders for K-Fold Cross Validation;
    -  [src](https://github.com/arianna011/Audio-ResiDual/tree/main/src): the main project module, which includes
          - **analyze_attention.py**: code to extract attention representantions from HTSAT audio encoder, perform PCA and store results on file
          - **residual.py**: implementantion of the ResiDual module and of its injection into HTSAT
          - **linear.py**: implementantion of a linear classifier on top of the frozen HTSAT encoder for the comparative evaluation
          - **training.py**: code to train ResiDual and log metrics with [Weights&Bias](https://wandb.ai/site/)
          - **evaluation.py**: code to evaluate the performance of the different CLAP versions on audio classification and visualize the results
