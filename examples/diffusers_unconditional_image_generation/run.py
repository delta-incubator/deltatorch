# Databricks notebook source
# MAGIC %pip install diffusers==0.16.1 accelerate>=0.18.0 torch==2.0.0 torchvision datasets  xformers tensorboard

# COMMAND ----------

# MAGIC !/databricks/python3/bin/python -m pip install diffusers==0.16.1 accelerate>=0.18.0 torch==2.0.0 torchvision datasets  xformers tensorboard git+https://github.com/mshtelma/deltatorch

# COMMAND ----------

# MAGIC !cd ../../ && /databricks/python3/bin/python -m pip install  -U .

# COMMAND ----------

# MAGIC !accelerate launch --mixed_precision="fp16" --multi_gpu --num_processes 8 train_unconditional.py \
# MAGIC   --train_data_dir="/dbfs/tmp/msh/huggan_pokemon.delta" \
# MAGIC   --resolution=64 \
# MAGIC   --center_crop \
# MAGIC   --random_flip \
# MAGIC   --output_dir="/dbfs/msh/deltatorch/diffusers/ddpm-ema-pokemon-64-v3" \
# MAGIC   --train_batch_size=96 \
# MAGIC   --num_epochs=100 \
# MAGIC   --gradient_accumulation_steps=1 \
# MAGIC   --use_ema \
# MAGIC   --learning_rate=1e-4 \
# MAGIC   --lr_warmup_steps=500 \
# MAGIC   --mixed_precision="fp16" \
# MAGIC  --enable_xformers_memory_efficient_attention \
# MAGIC   --logger tensorboard

# COMMAND ----------

from diffusers import DiffusionPipeline

generator = DiffusionPipeline.from_pretrained(
    "/dbfs/msh/deltatorch/diffusers/ddpm-ema-pokemon-64-v2"
).to("cuda")

# COMMAND ----------

generator().images[0]

# COMMAND ----------

generator().images[0]

# COMMAND ----------

generator().images[0]

# COMMAND ----------

generator().images[0]

# COMMAND ----------

generator().images[0]

# COMMAND ----------
