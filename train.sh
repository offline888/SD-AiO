git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .

To start, you must have a dataset containing triplets:

* Condition image - the input image to be transformed.
* Target image - the desired output image after transformation.
* Instruction - a text prompt describing the transformation from the condition image to the target image.

[kontext-community/relighting](https://huggingface.co/datasets/kontext-community/relighting) is a good example of such a dataset. If you are using such a dataset, you can use the command below to launch training:

```bash
accelerate launch train_dreambooth_lora_flux2_img2img.py \
  --pretrained_model_name_or_path=black-forest-labs/FLUX.2-dev  \
  --output_dir="flux2-i2i" \
  --dataset_name="kontext-community/relighting" \
  --image_column="output" --cond_image_column="file_name" --caption_column="instruction" \
  --do_fp8_training \
  --gradient_checkpointing \
  --remote_text_encoder \
  --cache_latents \
  --resolution=1024 \
  --train_batch_size=1 \
  --guidance_scale=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --optimizer="adamw" \
  --use_8bit_adam \
  --cache_latents \
  --learning_rate=1e-4 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=200 \
  --max_train_steps=1000 \
  --rank=16\
  --seed="0" 