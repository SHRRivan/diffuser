from diffusers import StableDiffusionImg2ImgPipeline
import torch, random, os, json
from PIL import Image
import glob

model_path = "models/AI-ModelScope/stable-diffusion-v1-5"
base_img_path = "img2img_base"
output_img_path = "img2img_aug"

os.makedirs(output_img_path, exist_ok=True)

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16).to("cuda")

lora_path = "weights/construction"
pipe.unet.load_lora_adapter(lora_path, weight_name="pytorch_lora_weights.safetensors")
pipe.enable_xformers_memory_efficient_attention()

supported_formats = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp')

weathers = [
    "early morning fog",
    "heavy overcast",
    "golden hour",
    "night floodlight",
    "rain",
    "crisp dawn light",
    "midday high sun with harsh shadows",
    "post-sunset blue hour",
    "thin high clouds filtering sunlight",
    "sun after brief summer shower",
    "cold winter haze",
    "strong crosswind with dust plumes",
    "snow flurries",
    "humid tropical glare",
    "drizzle with rainbow backdrop",
    ""
]
meters = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
workers = [
    "workers in red vests and yellow helmets",
    "night crew with LED armbands",
    "reflective-strip workers operating concrete vibrator",
    "welder with spatter-proof mask and leather apron",
    "crane operator in orange harness peering from cab",
    "survey crew in hi-vis coats setting up total station",
    "rebar workers with steel-toe boots and cut-resistant gloves",
    "road paver team in grey coveralls and ear defenders",
    "scaffold riggers wearing fall-arrest lanyards",
    "drill crew in dust masks and goggles near rock face",
    "traffic controllers with LED batons and stop-slow paddles",
    "electricians in arc-flash suits pulling cable trays",
    "concrete pump truck crew in waterproof overalls",
    "excavator spotter in bright green vest holding radio",
    "safety supervisor with white helmet and clipboard",
    ""
]
numbers = ["one", "two", "three"]
backgrounds = [
    "urban skyline",
    "mountainous terrain",
    "desert dunes",
    "coastal cliffs",
    "industrial port at sunrise",
    "highway interchange under construction",
    "wind farm on rolling hills",
    "solar panel array in rural flatland",
    "dam construction site in river valley",
    "railway viaduct cutting through forest",
    "suburban development zone with earthworks",
    "open-pit mine with terraced slopes",
    "airport runway extension site",
    "offshore wind turbine installation",
    "bridge construction over wide river"
]
pixels = [720, 1080]

def process_images_recursively(input_folder, output_folder, num_augmentations=5):
    """
    Args:
        input_folder: 
        output_folder: 
        num_augmentations: how many argumented-images will generated of each base image
    """
    image_files = []
    for format in supported_formats:
        pattern = os.path.join(input_folder, '**', format)
        image_files.extend(glob.glob(pattern, recursive=True))
    
    print(f"Found {len(image_files)} base images in total...")
    
    # processing single image
    for img_idx, img_path in enumerate(image_files):
        if img_idx >= 1:
            break
        try:
            base_img = Image.open(img_path).convert("RGB")
            
            relative_path = os.path.relpath(img_path, input_folder)
            img_dir = os.path.dirname(relative_path)
            img_base_name = os.path.splitext(os.path.basename(img_path))[0]
            
            output_subdir = os.path.join(output_folder, img_dir)
            os.makedirs(output_subdir, exist_ok=True)
            
            for aug_idx in range(num_augmentations):
                w, m, p, b, n, x = (random.choice(weathers),
                                 random.choice(meters),
                                 random.choice(workers),
                                 random.choice(backgrounds),
                                 random.choice(numbers),
                                 random.choice(pixels),
                                )
                prompt = (f"aerial view of {m} meter's construction site, {n} circular excavation pits in a row, "
                          f"red and white safety fence tape, {p}, "
                          f"{b} background, "
                          f"{w}, {x}P, photorealistic, sharp focus")
                
                neg = "cartoon, lowres, distorted fence, shifted excavation, watermark, text, "
                neg += "missing pits, extra pits, non-circular pits, buildings, trees, blue sky"
                
                # use both image index and augment to compute seed
                seed = img_idx * num_augmentations + aug_idx
                
                # generate
                img = pipe(prompt, negative_prompt=neg, image=base_img,
                           strength=0.6, guidance_scale=10, num_inference_steps=80,
                           generator=torch.Generator().manual_seed(seed)).images[0]
                
                output_filename = f"{img_base_name}_aug_{aug_idx:03d}.jpg"
                output_path = os.path.join(output_subdir, output_filename)
                img.save(output_path)
                
                print(f"  Generate {aug_idx + 1}/{num_augmentations}: {output_filename}")
            
            # save_generation_parameters(output_subdir, img_base_name, num_augmentations)
            
        except Exception as e:
            print(f"Some error when handling {img_path}: {e}")
            continue

def save_generation_parameters(output_dir, base_name, num_augmentations):
    """save generate param to JSON file"""
    params = {
        "base_image": base_name,
        "augmentations_count": num_augmentations,
        "parameters": {
            "weather_options": weathers,
            "meter_options": meters,
            "worker_options": workers,
            "number_options": numbers,
            "background_options": backgrounds,
            "pixel_options": pixels
        }
    }
    
    params_file = os.path.join(output_dir, f"{base_name}_generation_params.json")
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=2)


if __name__ == "__main__":
    process_images_recursively(base_img_path, output_img_path, num_augmentations=3)
    print("----------All images done!----------")
