from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
import os

model_path = "models/AI-ModelScope/stable-diffusion-v1-5"

generate_image_path = "GENERATED"
os.makedirs(generate_image_path, exist_ok=True)

pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

lora_path = "weights/construction"
pipe.unet.load_lora_adapter(lora_path, weight_name="pytorch_lora_weights.safetensors")
pipe.to("cuda")
pipe.enable_xformers_memory_efficient_attention()

# generate images by different prompt and seed
PROMPTS = [
    "An aerial view from 50 meters above, capturing a construction site with multiple yellow excavators digging into earthy terrain, surrounded by modern high-rise buildings and busy city streets, with workers in safety vests scattered around.",
    "An aerial view from 100 meters above, showing a partial construction zone with cranes and concrete forms, set against a backdrop of lush green mountains and a winding river, under clear daylight.",
    "An aerial view from 30 meters above, featuring a busy construction area with dump trucks and scaffolding, integrated into an urban landscape with skyscrapers, roads, and pedestrian traffic.",
    "An aerial view from 80 meters above, highlighting a construction project with orange cranes lifting materials, alongside a bustling city center with commercial buildings, parks, and moving vehicles.",
    "An aerial view from 120 meters above, featuring a construction zone with temporary fencing and blue tarps, set in a mixed environment of industrial buildings and natural foothills under cloudy skies.",
    "An aerial view from 60 meters above, depicting a construction site with concrete pouring and scaffolding, surrounded by a mountainous landscape with pine trees and a small lake.",
    "An aerial view from 180 meters above, showing a partial construction area with cranes and debris, integrated into a coastal city with ports, beaches, and high-rise apartments.",
    "An aerial view from 70 meters above, capturing a construction site with vibratory rollers and compactors, embedded in a suburban setting with schools, parking lots, and tree-lined streets.",
    "An aerial view from 50 meters above, showing a highway interchange under construction with multiple bridge pillars being erected, surrounded by flowing traffic on existing roads, with construction workers and cranes visible amidst the concrete structures.",
    "An aerial view from 130 meters above, showing a construction zone with drilling rigs and dump trucks, situated in a rural area with farmland, barns, and distant forests.",
    "An aerial view from 35 meters above, depicting a construction site with workers binding rebar and operating cranes, in an urban park setting with walking paths, benches, and city towers.",
    "An aerial view from 160 meters above, capturing a construction area with modular buildings and cranes, surrounded by a mix of old and new urban infrastructure, including bridges and railways.",
    "An aerial view from 45 meters above, showing a construction zone with concrete mixers and formwork, integrated into a bustling commercial district with shops, signage, and traffic lights.",
    "An aerial view from 120 meters above, capturing a residential complex construction in a suburban area, with half-completed apartment buildings, temporary access roads, and a backdrop of sprawling residential neighborhoods and green parks.",
    "An aerial view from 80 meters above, depicting a river embankment reinforcement project, with excavators and dump trucks working along the riverbank, surrounded by natural vegetation and a city skyline in the distance.",
    "An aerial view from 150 meters above, showing a tunnel entrance construction on a mountainside, with drilling equipment and support structures being installed, surrounded by dense forest and winding access roads.",
    "A drone's perspective from 60 meters, showing a fenced-off construction area for an underground utility project in the center of a large urban park. People are jogging and walking dogs on paths nearby, with trees and a playground in the scene.",
    "An aerial view from 60 meters above, featuring an urban subway station construction site with deep excavation, surrounded by commercial buildings, busy streets with vehicles and pedestrians, and safety fencing marking the perimeter.",
    "A wide-angle aerial view from 200 meters, showing a large-scale land reclamation and construction project at a coastal port. Dump trucks move earth against a backdrop of cargo ships, cranes, and the open sea. Hazy coastal atmosphere.",
    "An aerial view from 200 meters above, capturing a large-scale industrial park development with multiple factory buildings in various stages of completion, access roads, and open areas of turned earth against a landscape of mixed fields and distant mountains.",
    "A drone view from 45 meters, focused on the reinforced concrete entrance of a tunnel under construction. Construction vehicles enter and exit the dark tunnel mouth, with the steep, rocky face of a hill covered in greenery above.",
    "An aerial view from 45 meters above, showing a bridge construction over a river, with temporary support structures and workers on the deck, with water flow underneath and city infrastructure on both banks.",
    "An aerial view from 100 meters above, depicting a slope stabilization project on a hillside, with earth-moving machinery and workers installing retaining structures, surrounded by residential buildings below and natural terrain above.",
    "An aerial view from 80 meters, showing a partially demolished building surrounded by protective netting. Dust clouds float around the wrecking ball. Intact residential buildings are close on all sides, highlighting the tight urban space.",
    "An oblique drone view from 70 meters, emphasizing the complex earthworks and retaining walls being constructed on a hillside for a new building. The slope and terracing are clearly visible against a backdrop of more distant mountains.",
    "An aerial view from 180 meters above, showing a wind farm construction with foundations being poured access roads winding up the mountain, and completed turbines in the background against a cloudy sky.",
    "A drone view from 100 meters, positioned to show the new bridge deck under construction above active railway tracks. A train is passing underneath, providing a sense of scale and operational complexity.",
    "An aerial view from 70 meters above, capturing a commercial building construction in a business district, with steel framework rising above ground level, surrounded by existing office towers, roads with traffic, and pedestrian walkways.",
    "An aerial photo from 130 meters, framing the construction site in the foreground with a famous city landmark or a beautiful natural feature (e.g., a lake, a mountain range) visible in the background, showcasing the context of the build.",
    "A drone's eye view from 60 meters, following the designated emergency vehicle access road through the construction site, ensuring it is clear of obstructions. The path leads from the main gate to the heart of the site.",
    "A series of descriptive prompts for a time-lapse: A drone observes from a safe distance of 120 meters as a large mobile crane is assembled on site, with smaller cranes helping to lift its components into place.",
    "An aerial view from 130 meters above, showing a water treatment plant expansion project with complex piping and concrete structures, set against a natural landscape with trees and a nearby river.",
    "An aerial view from 95 meters, capturing the movement of personnel as one shift ends and another begins. Clusters of workers near the gates, increased activity in the parking lot, and a change in machinery operation patterns.",
    "An aerial perspective from 110 meters, showcasing environmental measures like dust suppression misting systems in operation, and designated areas for recycling construction waste.",
    "A drone looking straight down into a deep, multi-level excavation pit from 80 meters above. The different stages of shoring and dewatering pumps are visible, with workers appearing very small at the bottom.",
    "An aerial view from 35 meters above, depicting a road widening project with asphalt paving machinery at work, one lane open to traffic, and construction signs guiding vehicles through the work zone.",
    "Aerial view from 60m of a deep circular excavation with interlocking stone block walls. Workers in red vests survey the pit. Grassy terrain with safety fencing.",
    "Low-angle aerial drone view of an active construction site with circular excavation pits, workers in red uniforms and yellow hard hats operating near the pits' edges, scattered tools, yellow caution tape, and green grass in the background.",
    "High-angle drone shot of a construction site featuring deep circular pits with visible internal cavities, mounds of rocks and dirt, workers in red overalls, and patches of water reflecting the sky, surrounded by a grassy field.",
    "Drone perspective from a medium altitude capturing multiple rocky circular pits at a construction site, with red-clad workers using equipment, yellow safety barriers, muddy ground, and overcast sky above the green landscape.",
    "An aerial view from 160 meters above, showing a port expansion project with land reclamation activities, cranes operating near the waterfront, and cargo ships anchored nearby in the harbor.",
    "Aerial view of an elliptical construction pit with rocky edges, surrounded by dirt piles. Workers in red uniforms operate near the pit, with yellow safety lines and excavators in a modern city setting.",
    "An aerial view from 90 meters above, capturing a school campus construction with multiple buildings underway, playground areas taking shape, and residential neighborhoods visible beyond the site boundaries.",
    "Overhead view from 100m showing an elliptical excavation site with accumulated rainwater. Construction team in red operates excavators, surrounded by mixed urban and natural landscape.",
    "Aerial perspective from 150m capturing an oval construction pit with machinery tracks. Construction crew in red uniforms coordinates operations as trucks deliver materials through wooded terrain.",
    "Aerial drone view of a deep elliptical excavation pit with rocky soil layers. Construction workers in safety vests operate near the pit, surrounded by yellow caution tape and excavators, with a modern city skyline in the background.",
    "High-angle perspective from 80m height, showing a circular construction site with a deep pit and concrete foundations. A construction crew is working, with cranes and temporary structures visible, set against distant mountains.",
    "Drone photography of an oval-shaped quarry pit with accumulated groundwater. Safety officers in reflective vests are conducting a survey, with construction vehicles and a forested area surrounding the site.",
    "An aerial view from 140 meters above, showing a mixed-use development project combining residential and commercial structures, with construction vehicles moving between buildings, all set within an urban context with surrounding city infrastructure."
    "Top-down drone image of a complex elliptical pit with stone and debris. Construction activities are ongoing with small figures of workers and vehicles, situated in a landscape transitioning between natural terrain and new infrastructure."
]


if __name__ == '__main__':
    for i, prompt in enumerate(PROMPTS):
        for seed in range(50):
            generator = torch.Generator("cuda").manual_seed(seed)
            image = pipe(prompt=prompt, height=640, width=640, num_inference_steps=80, generator=generator).images[0]
            image.save(f"{generate_image_path}/PROMPT_{i}-SEED_{seed}.jpg")
