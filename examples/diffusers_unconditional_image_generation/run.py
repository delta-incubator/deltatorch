# Databricks notebook source
# MAGIC %pip install diffusers==0.16.1 accelerate>=0.18.0 torch==2.0.0 torchvision datasets  xformers tensorboard

# COMMAND ----------

!/databricks/python3/bin/python -m pip install diffusers==0.16.1 accelerate>=0.18.0 torch==2.0.0 torchvision datasets  xformers tensorboard git+https://github.com/mshtelma/deltatorch

# COMMAND ----------

!cd ../../ && /databricks/python3/bin/python -m pip install  -U .

# COMMAND ----------

!accelerate launch --mixed_precision="fp16" --multi_gpu --num_processes 4 train_unconditional.py \
  --train_data_dir="/dbfs/tmp/msh/huggan_pokemon.delta" \
  --resolution=64 \
  --center_crop \
  --random_flip \
  --output_dir="/dbfs/msh/deltatorch/diffusers/ddpm-ema-pokemon-64-v2" \
  --train_batch_size=16 \
  --num_epochs=100 \
  --gradient_accumulation_steps=1 \
  --use_ema \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --mixed_precision="fp16" \
  --enable_xformers_memory_efficient_attention \
  --logger tensorboard

# COMMAND ----------

from diffusers import DiffusionPipeline

generator = DiffusionPipeline.from_pretrained("/dbfs/msh/deltatorch/diffusers/ddpm-ema-pokemon-64-v2").to("cuda")

# COMMAND ----------

generator().images[0]

# COMMAND ----------

generator().images[0]

# COMMAND ----------


