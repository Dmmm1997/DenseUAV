# Parameter Introduction

### Data-Related Parameters
- `--name`: Experiment name, used for saving models and parameters under `checkpoints/`, facilitating management and tracking of different experimental results.
- `--data_dir`: Directory path for training data.
- `--num_worker`: Number of worker threads used for data loading, affecting the parallelism and efficiency of data preprocessing.
- `--pad`: Amount of padding for input data. Please distinguish this from the `--pad` in Position Shifting.
- `--h, --w`: Height and width of the input images.
- `--rr`: Random rotation applied to one or more views to enhance data diversity.
- `--ra`: Random affine transformation applied to one or more views to enhance data diversity.
- `--re`: Random occlusion applied to one or more views to enhance data diversity.
- `--cj`: Color jitter applied to one or more views to enhance data diversity.
- `--erasing_p`: Probability of random occlusion, controlling the proportion of randomly occluded areas in the images.

### Training-Related Parameters
- `--warm_epoch`: Warm-up phase, setting the learning rate to gradually increase over the first `K` epochs.
- `--lr`: Learning rate.
- `--DA`: Whether to use color data augmentation.
- `--droprate`: Dropout rate.
- `--autocast`: Whether to use mixed precision training.
- `--load_from`: Path to the pre-loaded checkpoint for restoring the model from a previous training state.
- `--gpu_ids`: Specification of the GPU devices used, supporting multi-GPU configurations for flexible training environments.
- `--batchsize`: Number of samples per training step.

### Model-Related Parameters
- `--block`: Number of ClassBlocks in the model.
- `--cls_loss`: Type of loss function for Representation Learning. Various preset or custom losses can be used, with `CELoss` as the default.
- `--feature_loss`: Type of loss function for Metric Learning. Various preset or custom losses can be used, with no loss applied by default.
- `--kl_loss`: Type of loss function for Mutual Learning. Various preset or custom losses can be used, with no loss applied by default.
- `--num_bottleneck`: Dimensionality of feature embeddings.
- `--backbone`: Backbone architecture used. Various preset or custom backbones can be selected, with `cvt13` as the default.
- `--head`: Head architecture used. Various preset or custom heads can be selected, with `FSRA_CNN` as the default.
- `--head_pool`: Type of pooling used in the head, with various preset or custom pooling methods available, defaulting to `max pooling`.
