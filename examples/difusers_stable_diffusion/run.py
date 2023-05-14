# Databricks notebook source
!/databricks/python3/bin/python -m pip install diffusers==0.16.1 accelerate>=0.18.0 torch==2.0.0 torchvision datasets  xformers tensorboard git+https://github.com/mshtelma/deltatorch ftfy tensorboard Jinja2

# COMMAND ----------

!/databricks/python3/bin/python -m pip install  git+https://github.com/huggingface/diffusers.git


# COMMAND ----------

!cd ../../ && /databricks/python3/bin/python -m pip install  -U .

# COMMAND ----------

# MAGIC %pip install accelerate>=0.18.0

# COMMAND ----------

!accelerate launch --mixed_precision fp16 --multi_gpu  --num_processes 8 train_text_to_image_lora.py \
--pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" \
--dataset_name="lambdalabs/pokemon-blip-captions" \
--caption_column="text" \
--resolution=512 \
--random_flip \
--train_batch_size=5 \
--num_train_epochs=100 \
--checkpointing_steps=1000 \
--learning_rate=1e-04 --lr_scheduler="constant" \
--lr_warmup_steps=0 \
--seed=42 \
--output_dir="/dbfs/msh/deltatorch/diffusers/sd-pokemon-model-lora" \
--report_to="tensorboard" \
--validation_prompt="cute dragon creature"

# COMMAND ----------

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

model_id = "stabilityai/stable-diffusion-2-1"

# Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt).images[0]
image

# COMMAND ----------

# MAGIC %sh accelerate launch --mixed_precision fp16 --multi_gpu  --num_processes 8 train_text_to_image.py \
# MAGIC --pretrained_model_name_or_path CompVis/stable-diffusion-v1-4 \
# MAGIC --dataset_name lambdalabs/pokemon-blip-captions \
# MAGIC --use_ema \
# MAGIC --resolution 512 \
# MAGIC --center_crop \
# MAGIC --random_flip \
# MAGIC --train_batch_size 1 \
# MAGIC --gradient_accumulation_steps 4 \
# MAGIC --gradient_checkpointing \
# MAGIC --max_train_steps 15000 \
# MAGIC --enable_xformers_memory_efficient_attention \
# MAGIC --learning_rate=1e-05 \
# MAGIC --max_grad_norm=1 \
# MAGIC --lr_scheduler="constant" --lr_warmup_steps=0 \
# MAGIC --output_dir /dbfs/msh/deltatorch/diffusers/sd-pokemon-model

# COMMAND ----------


