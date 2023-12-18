# Scripts description
1. **preprocess.py**:
    Training dataset augmentation offline. Support up-sampling strategy based on positive and negative.
2. **rooftop_dataset.py**:Create training, validation and test dataset in pytorch format.
3. **train.py**: Training of rooftop extraction model.
4. **predict.py**: Batch prediction based on ensemble strategy and expansion prediction method.
5. **predict_trt.py**: Batch prediction based on TensorRT.
6. **evaluation.py**: Evaluation of dataset.
7. **deeplab_resnet.py**: Deeplabv3+ model based on resnet backbone.
8. **deeplab_xception.py**: Deeplabv3+ model based on xception backbone.
9. **ensemble_config.json**: Configuration of ensemble model. (The path of checkpoints)

Citation: Zhang, Z., Qian, Z., Zhong, T., Chen, M., Zhang, K., Yang, Y., Zhu, R., Zhang, F., Zhang, H., Zhou, F. and Yu, J., 2022. Vectorized rooftop area data for 90 cities in China. Scientific Data, 9(1), pp.1-12.
