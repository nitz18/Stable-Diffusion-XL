import torch
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
pipe = pipe.to("cuda") 
pipe.load_lora_weights("weights_file_path_here")

refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)
refiner = refiner.to("cuda")  

prompts = [
    ("A picture of Ronaldo showing the thumbs up gesture on top of a hill", "hill"),
    ("A picture of Ronaldo showing thumbs up gesture while holding an orange juice box in his right hand", "juice"),
    ("A picture of Ronaldo showing the thumbs up gesture while swimming", "swim"),
    ("A picture of Ronaldo showing the thumbs up gesture in a press conference", "press"),
    ("A picture of Ronaldo showing the thumbs up gesture on the moon", "moon")
]

generator = torch.Generator("cuda").manual_seed(0)

for prompt, suffix in prompts:
    # Run inference
    image = pipe(prompt=prompt, output_type="latent", generator=generator, num_inference_steps=50).images[0]
    image = refiner(prompt=prompt, image=image[None, :], generator=generator).images[0]
    image.save(f"refined_{suffix}.png")
