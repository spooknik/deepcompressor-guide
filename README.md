# Deepcompressor Configs

**VERY WIP**

This is a repo that aims to document the process for creating SVDQuants using [Deepcompressor](https://github.com/nunchaku-tech/deepcompressor). The Nunchaku team did an excellent job with the project, but I found a practical guide was lacking.

0. Considerations

- SVDQaunts take a lot of GPU compute time. For a Flux.1 Dev model it can take around 60 hours of compute time to make int4 and nvfp4 versions.
- A lot VRAM, it can be done on 48GB, but it's about twice or 3 times as fast on a card with 96GB because we can use large batch sizes. 

Step 1: Evaluation Baselines Preparation

Deepcompressor will sample the unquantized version of the model to have a reference (baseline) to make an evaluation against the quantized. This is a good thing because we'll know objectively how the quantized version of the model performs. 

I limited the samples to 256 images, the default is 5000, Nunchaku's examples were using both 256 and 1024. The higher the number the more accurate the comparison. But the longer the compute time. 256 images takes around 1.5 to 2 hours. 



```bash
poetry run python -m deepcompressor.app.diffusion.ptq configs/models/[CONFIG] --output-dirname reference
```



Step 2: Calibration Dataset Preparation
```bash
poetry run python -m deepcompressor.app.diffusion.dataset.collect.calib \
    configs/models/[CONFIG] configs/collect/qdiff.yaml
```



Step 3: Model Quantization

Now the long part (around 24 hours).  

```bash
poetry run python -m deepcompressor.app.diffusion.ptq \
    configs/models/CenKreChroA40.yaml configs/svdquant/int4.yaml configs/svdquant/fast.yaml \
    --eval-benchmarks MJHQ \
    --eval-num-samples 256 \
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
Step 4: Deployment

```
poetry run python -m deepcompressor.backend.nunchaku.convert \
  --quant-path /PATH/TO/CHECKPOINT/DIR \
  --output-root /PATH/TO/OUTPUT/ROOT \
  --model-name MODEL_NAME
```
Then merge. You need a config.json and comfy_config.json for the model. 
https://nunchaku.tech/docs/nunchaku/python_api/nunchaku.merge_safetensors.html
```
poetry run python -m nunchaku.merge_safetensors -i /workspace/deepcompressor/outputs/[PROJECT}-o  /workspace/deepcompressor/outputs/merged
```
