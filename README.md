# deepcompressor-configs
A repo for Deepcompressor configs

Step 1: Evaluation Baselines Preparation
```bash
poetry run python -m deepcompressor.app.diffusion.ptq configs/models/CenKreChroA40.yaml --output-dirname reference
```

Step 2: Calibration Dataset Preparation
```bash
poetry run python -m deepcompressor.app.diffusion.dataset.collect.calib \
    configs/models/CenKreChroA40.yaml configs/collect/qdiff.yaml
```

Step 3: Model Quantization
```bash
poetry run python -m deepcompressor.app.diffusion.ptq \
    configs/models/CenKreChroA40.yaml configs/svdquant/int4.yaml configs/svdquant/fast.yaml \
    --eval-benchmarks MJHQ \
    --eval-num-samples 1024 \
    --save-model checkpoint/int4 \
    --smooth-proj-outputs-device cuda \
    --smooth-attn-outputs-device cuda \
    --wgts-low-rank-outputs-device cuda \
    --wgts-calib-range-outputs-device cuda \
    --ipts-calib-range-outputs-device cuda \
    --opts-calib-range-outputs-device cuda \
    --text-wgts-calib-range-outputs-device cuda \
    --text-ipts-calib-range-outputs-device cuda \
    --text-opts-calib-range-outputs-device cuda \
    --text-reorder-outputs-device cuda \
    --enable-smooth true \   
    --enable-cache true \
    --cache-root /workspace/deepcompressor/cache
```
