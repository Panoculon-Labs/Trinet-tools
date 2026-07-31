# HITNet ONNX models (not committed)

`scripts/stereo_depth_neural.py` expects `hitnet_mb_720x1280.onnx` here
(Middlebury-trained HITNet, fixed 1280x720 input, `[1,6,H,W]` = RGB left +
RGB right in 0..1, disparity out `[1,H,W,1]`).

Source: PINTO model zoo, model 142_HITNET —

    curl -O https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo/142_HITNET/resources.tar.gz
    tar -xzf resources.tar.gz \
        middlebury_d400/saved_model_720x1280/model_float32.onnx \
        middlebury_d400/saved_model_480x640/model_float32.onnx
    cp middlebury_d400/saved_model_720x1280/model_float32.onnx hitnet_mb_720x1280.onnx
    cp middlebury_d400/saved_model_480x640/model_float32.onnx  hitnet_mb_480x640.onnx

The `.onnx` files are gitignored (binary weights, ~8 MB each; upstream is the
source of truth). HITNet: Tankovich et al., CVPR 2021 (arXiv:2007.12140).

GPU inference needs onnxruntime-gpu plus the pip CUDA-13 runtime wheels:

    uv pip install onnxruntime-gpu nvidia-cuda-runtime nvidia-cublas \
        nvidia-curand nvidia-cufft nvidia-cudnn-cu13

(~0.3 s/frame on an RTX 5060 Ti at 720p; CPU fallback works, ~12x slower.)
