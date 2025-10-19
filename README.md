# Deepcompressor Configs

**VERY WIP**

This is a repo that aims to document the process for creating SVDQuants using [Deepcompressor](https://github.com/nunchaku-tech/deepcompressor). The Nunchaku team did an excellent job with the project, but I found a practical guide was lacking.

## 0. Considerations

- SVDQaunts take a lot of compute time. For a Flux.1 Dev model it can take around 24-28 per quant.
- You cannot do this on a consumer GPU, you need a lot of VRAM, it can be done on 48GB, but it's slow and low quality. Ideally you want 80GB or even better 96GB. 
    - I have had good luck with the RTX 6000 Pro cards, they are 96gb and much cheaper than H100.
- Cloud GPU providers like Runpod or Vast.ai are good choices. Keep in mind the cost, 24 hours x hourly price of your cloud instance. 

## 1: Setup

I have made a setup script to prepare the environment for deepcompressor.

So in bash run:

```bash
cd /workspace # or where you wish to run it
wget script.sh
chmod +x script.sh
./script.sh
```
Let it run, it takes around 5 minutes depending on network speed. Towards the end it will ask if you want to login with HuggingFace and there you just need to paste your API key. 

Folder structure

workspace


## 2: Evaluation Baselines Preparation

Deepcompressor will sample the BF16/F16 version of the model to have a reference (baseline) to make an evaluation against the quantized, you can skip this to save around 2-3 hours, but then you'll not really have an objective measurement of how good your quant is. 

I limited the samples to 256 images, the default is 5000, Nunchaku's examples were using both 256 and 1024. The higher the number the more accurate the comparison. 128 also works, just the result might be even less accurate. 


```bash
poetry run python -m deepcompressor.app.diffusion.ptq configs/models/[CONFIG] --output-dirname reference
```


## 3: Calibration Dataset Preparation


```bash
poetry run python -m deepcompressor.app.diffusion.dataset.collect.calib \
    configs/models/[CONFIG] configs/collect/qdiff.yaml
```



## 4.

Now the long part (around 20 hours).  

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
5. Deployment

```
poetry run python -m deepcompressor.backend.nunchaku.convert \
  --quant-path /PATH/TO/CHECKPOINT/DIR \
  --output-root /PATH/TO/OUTPUT/ROOT \
  --model-name MODEL_NAME


```

6. Merging
Then merge. You need a config.json and comfy_config.json for the model. 

