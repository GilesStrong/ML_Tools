# HEPML Tools
A collection of modules, functions, and wrappers for with Keras designed for application of deep learning to high energy physics.
This adapts pretty frequently depending on the needs of my research, but I'm hoping to stabalise things a bit, improve user friendliness, and add documentation and examples with the aim of it being more widely used by the physics community.

## Distinguishing Characteristics
- Use with large datasets: HEP data can become quite large, the fold* methods expect to be provided with HDF5 files of the data split into smaller folds. These are then loaded sequentially into memory. Perhaps in future they may be changed to Keras data generators.
- Handling of sample weights: HEP events are normally accompanied by weight characterising the acceptance and production cross-section of that particular event, or to flatten some distribution. Relevant methods here can take account of these weights.
- Inclusion of recent deep learning techniques and practices, including:
    - Dynamic learning rate, momentum, $\beta_1$: 
        - Cyclical, [Smith, 2015](https://arxiv.org/abs/1506.01186)
        - Cosine annealed [Loschilov & Hutte, 2016](https://arxiv.org/abs/1608.03983)
        - 1-cycle, [Smith, 2018](https://arxiv.org/abs/1803.09820)
    - HEP-specific data augmentation
    - Stochastic Weight Averaging [Izmailov, et al., 2018](https://arxiv.org/abs/1803.05407)
    - Learning Rate Finders, [Smith, 2015](https://arxiv.org/abs/1506.01186)
    - Entity embedding of categorical features, [Guo & Berkhahn, 2016](https://arxiv.org/abs/1604.06737)
- Easy training and inference of ensembles of models. 
- Various plotting functions, most of which account for sample weights
