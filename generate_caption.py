import json
import torch

from pathlib import Path
from tqdm import tqdm

from modelscope import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
"""
Maybe encounter error like: ImportError: cannot import name 'GELUPytorchTanh' from 'transformers.activations' 
(/home/lab314a/.conda/envs/modelscope/lib/python3.10/site-packages/transformers/activations.py)
And just add `PytorchGELUTanh = GELUTanh` after class GELUTanh
"""

ROOT_FOLDER = Path("datasets/small_construction")
MODEL_PATH = Path("models/Qwen/Qwen2___5-VL-7B-Instruct-AWQ")

METADATA = ROOT_FOLDER / "metadata.jsonl"

# 加载模型
print(f"Loading {MODEL_PATH} ...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
device = next(model.parameters()).device

# 收集待处理图片
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
all_imgs = [p for p in ROOT_FOLDER.rglob("*") if p.suffix.lower() in IMG_EXTS]

# 如果 metadata.jsonl 已存在，读取已完成的文件名，避免重复
done = set()
if METADATA.exists():
    with METADATA.open("r", encoding="utf-8") as f:
        for line in f:
            done.add(json.loads(line)["file_name"])

remaining = [img for img in all_imgs if img.name not in done]
print(f"Total images: {len(all_imgs)},  already done: {len(done)},  remaining: {len(remaining)}")

# set prompt, different prompt will generate different description of image
PROMPT = """
You are a professional image annotation assistant specializing in construction site imagery. Please generate a detailed English description for each image, which will be used to train a Stable Diffusion model. The images are all from construction sites. Please describe them from a professional construction perspective, focusing on the following aspects in a structured manner:

1.  **Scene Overview & Key Elements**: Identify the main setting (e.g., residential building, subway station) and the primary construction activities or key components (e.g., concrete pouring, steel bar binding, template installation, mechanical and electrical pre-embedment).
2.  **Construction Details & Specifications**: Describe specific construction practices, material specifications, and technical details. For example:
    *   If it involves formwork, describe its type (e.g., circular formwork), installation method, and support system (e.g., full framing scaffold with internal and external diagonal bracing).
    *   If it involves masonry or finishes, describe the process (e.g., layered process for stone plate paving: cement mortar, leveling, pure cement slurry) and material specifications (e.g., sintered porous brick specifications 240 * 190 * 90mm).
    *   Mention specific technical requirements, such as waterproof detailing at pipe penetrations through roofs: "waterproofing head secured with metal band, sealant applied at the top opening, waterproofing height ≥25cm, additional waterproof layer at pipe base".
3.  **Construction Subjects & Environment**: Clearly describe the construction entities (e.g., "a concrete vibrator," "a tower crane"), personnel roles (e.g., "workers installing rebar"), and the surrounding environment (e.g., "indoor rough work phase," "outdoor foundation construction site").
4.  **Visual Characteristics**: Accurately depict visual elements such as composition, perspective (e.g., close-up on a detail, overall view of the site), lighting conditions, and colors/textures of materials.
5.  **Must follow the rule:
    *   Output only the English description; no Chinese explanations.
    *   Do not start with phrases like “this image shows/depict/display”; describe the scene directly like 1."An aerial(eye-level, surveillance camera, nighttime, close-up or any else) view of" or 2."A xxx contruction site", or 3.someother style like this will be ok.
    *   One single paragraph per image, no line breaks.
6.  **Output example:
    *   An aerial view of a construction worker in a yellow hard hat operates an orange excavator digging a deep foundation trench beside a pile of steel rebar under bright midday sunlight.

Please ensure the descriptions are accurate, professional, rich in detail, and suitable for training a visual generation model.
"""

# PROMPT = """
# You are a professional image annotation assistant for construction site imagery. Generate a concise English description for each image, suitable for training a Stable Diffusion model. The description must be within 128 tokens.

# **Focus on these key aspects in a single, fluent paragraph:**
# 1.  **Main Scene & Key Elements**: The primary setting (e.g., residential high-rise, tunnel interior) and the most prominent construction activity or key components (e.g., excavator digging, workers installing rebar).
# 2.  **Primary Action or State**: The central action (e.g., crane lifting, concrete pouring) or notable condition.
# 3.  **Salient Visual Characteristics**: Notable perspective (e.g., aerial, eye-level), lighting (e.g., sunny, dusk), or striking colors/textures.

# **Crucial Instructions:**
# *   **Be concise and direct.** Prioritize the most important information.
# *   **Do not** start with phrases like "This image shows...". Begin directly with the scene description (e.g., "An aerial view of...", "A close-up of...").
# *   **Avoid** exhaustive technical specifications and lengthy lists of minor details. Mention only the most distinctive or critical ones.
# *   Output only the English description, nothing else.

# **Example Output:**
# An aerial view of a large construction site with an orange excavator digging a foundation trench near a pile of steel rebar under clear blue skies.
# """

def describe_image(image_path: Path) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path.absolute())},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.5,
        top_k=30,
        top_p=0.8,
        repetition_penalty=1.05,
    )
    generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


if __name__ == '__main__':
    with METADATA.open("a", encoding="utf-8") as f_out:
        for img_path in tqdm(remaining, desc="Processing"):
            try:
                desc = describe_image(img_path).strip()
            except Exception as e:
                print(f"[WARN] Failed on {img_path}: {e}")
                continue
            record = {"file_name": img_path.name, "text": desc}
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()

    print("Done")