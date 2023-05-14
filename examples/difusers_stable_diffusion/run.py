# Databricks notebook source
# MAGIC !/databricks/python3/bin/python -m pip install diffusers==0.16.1 accelerate>=0.18.0 torch==2.0.0 torchvision datasets  xformers tensorboard git+https://github.com/mshtelma/deltatorch ftfy tensorboard Jinja2

# COMMAND ----------

# MAGIC !/databricks/python3/bin/python -m pip install  git+https://github.com/huggingface/diffusers.git


# COMMAND ----------

# MAGIC !cd ../../ && /databricks/python3/bin/python -m pip install  -U .

# COMMAND ----------

# MAGIC %pip install accelerate>=0.18.0

# COMMAND ----------

# MAGIC !accelerate launch --mixed_precision fp16 --multi_gpu  --num_processes 8 train_text_to_image_lora.py \
# MAGIC --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
# MAGIC --train_data_dir="/dbfs/tmp/msh/lambdalabs_pokemon_blip_captions.delta" \
# MAGIC --resolution=512 \
# MAGIC --random_flip \
# MAGIC --train_batch_size=5 \
# MAGIC --num_train_epochs=100 \
# MAGIC --checkpointing_steps=1000 \
# MAGIC --learning_rate=1e-04 --lr_scheduler="constant" \
# MAGIC --lr_warmup_steps=0 \
# MAGIC --seed=42 \
# MAGIC --output_dir="/dbfs/msh/deltatorch/diffusers/sd-pokemon-model-lora-v2" \
# MAGIC --report_to="tensorboard" \
# MAGIC --validation_prompt="cute dragon creature"

# COMMAND ----------

from diffusers import StableDiffusionPipeline
import torch

model_path = "/dbfs/msh/deltatorch/diffusers/sd-pokemon-model-lora-v2/"

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
)
pipe.unet.load_attn_procs(model_path)
pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
image

# COMMAND ----------

image

# COMMAND ----------
