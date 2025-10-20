# Deepcompressor Configs

## ⚠️*Under Construction* ⚠️

This is a repo that aims to document the process for creating SVDQuants using [Deepcompressor](https://github.com/nunchaku-tech/deepcompressor). The Nunchaku team did an excellent job with the project, but I found a practical guide was lacking.

Here is [a link to my HuggingFace](https://huggingface.co/spooknik) with SVDQuants I have already prepared.

If you find my work useful:

<a href='https://ko-fi.com/B0B21MPRDT' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://storage.ko-fi.com/cdn/kofi6.png?v=6' border='0' alt='Buy Me a Coffee at ko-fi.com' /></a>

## 0. Considerations

- SVDQaunts take a lot of compute time. For a Flux.1 Dev model it can take around 24-28 per quant.
- Currently only Flux.1, SANA and PixArt are supported. It's expected Qwen and WAN will be released at some point.
- You cannot do this on a consumer GPU, you need a lot of VRAM, it can be done on 48GB, but it's slow and low quality. Ideally you want 80GB or even better 96GB. 
    - I have had good luck with the RTX 6000 Pro cards, they are 96gb and much cheaper than H100.
    - Don't skimp on weak CPU, single core performance matters a lot. 
- Cloud GPU providers like Runpod or Vast.ai are good choices. Keep in mind the cost, 24 hours x hourly price of your cloud instance. 

## 1.1: Setup Environment

I have made a setup script to prepare the environment for deepcompressor.

In Bash run:

```bash
cd /workspace # or where you wish to run it
wget https://raw.githubusercontent.com/spooknik/deepcompressor-guide/refs/heads/main/install_deepcompressor.sh
chmod +x install_deepcompressor.sh
./install_deepcompressor.sh
```
Let it run, it takes around 10 minutes depending on network speed. Towards the end it will ask if you want to login with HuggingFace and there you just need to paste your API key.

## 1.2: Setup Environment



## 2: Evaluation Baselines Preparation

Deepcompressor will sample the BF16/F16 version of the model to have a reference (baseline) to make an evaluation against the quantized, you can skip this to save around 2-3 hours, but then you'll not really have an objective measurement of how good your quant is. 

I limited the samples to 256 images, the default is 5000, Nunchaku's examples were using both 256 and 1024. The higher the number the more accurate the comparison. 128 also works, just the result might be even less accurate. 


```bash
poetry run python -m deepcompressor.app.diffusion.ptq configs/models/your-model.yaml --output-dirname reference
```


## 3: Calibration Dataset Preparation


```bash
poetry run python -m deepcompressor.app.diffusion.dataset.collect.calib \
    configs/models/your-model.yaml configs/collect/qdiff.yaml
```



## 4.

Now the long part (around 20 hours).  

```bash
poetry run python -m deepcompressor.app.diffusion.ptq \
    configs/models/your-model.yaml configs/svdquant/int4.yaml configs/svdquant/fast.yaml \
    --eval-benchmarks MJHQ \
    --eval-num-samples 256 \
    --save-model output/quantized
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

