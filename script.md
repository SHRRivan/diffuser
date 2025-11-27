# image to text by deepseek

```text
请根据这三张图片生成九段英文的prompt，我需要利用生成好的prompt去生成和给出图像类似的无人机视角下施工现场图片，所以需要你对施工场景有全面、多样地描述，尽可能在保持与图中相同施工场景下提供更多样的高度、背景、视角等因素，在不超过CLIP模型77token的限制下尽可能丰富。这三张图片的施工核心因素在于椭圆形的、具有一定深度的、内部空洞的、周围堆积了一定石头/泥土的石坑/土坑的场景，所生成的prompt必须基于这个场景。在真实的业务场景下，无人机会以不同的高度30-200，在现代化城市、山林等各种场景下巡检，这涉及到不同的背景；另外，施工场地可以是生成图片的一个部分或者整体，它应该富含现代化城市、山林、施工机械、施工人员等一种或多种额外的背景要素。生成的prompt以python列表形式提供。
```

# generate prompt for SD model by Kimi
```bash
这是我训练stable_diffusion时的文本标签文件，现在我需要你帮我基于训练时候的描述设计一系列英文prompt，要求给出的prompt围绕无人机视角（aerial view）下的施工场景图，生成的图片我要用来做无人机巡检场景下施工现场目标检测的训练集。在真实的业务场景下，无人机会以不同的高度30-200，在现代化城市、山林等各种场景下巡检；另外，施工场地应该只是生成图片的一个部分而不是整体，它应该富含现代化城市、山林、施工机械、施工人员等一种或多种额外的要素。
```

# fine-tune the model with lora
```bash
./docker_run.sh accelerate launch --mixed_precision="bf16" examples/text_to_image/train_text_to_image_lora.py \
  --pretrained_model_name_or_path="./models/AI-ModelScope/stable-diffusion-v1-5" \
  --train_data_dir="./datasets/small_construction" \
  --image_column="image" \
  --caption_column="text" \
  --resolution=512 \
  --center_crop \
  --random_flip \
  --train_batch_size=4 \
  --max_train_steps=1000 \
  --rank=128 \
  --learning_rate=1e-04 \
  --lr_scheduler="cosine" \
  --enable_xformers_memory_efficient_attention \
  --checkpointing_steps=500 \
  --output_dir="./weights/small_construction" \
  --seed=17
```

# generate caption and image
```bash
conda activate modelscope
python generate_caption.py
```
```bash
./docker_run.sh python generate_image.py
```

# img2img usage
```bash
./docker_run.sh python generate_image_by_image.py
```