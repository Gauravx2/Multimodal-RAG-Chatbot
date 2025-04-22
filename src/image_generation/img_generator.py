from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
from PIL import Image
import os
from typing import Optional, Dict, Any
from tqdm import tqdm
import time

class BiomedicalImageGenerator:
    def __init__(self):
        """
        Initialize the biomedical image generator
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        try:
            print("Loading model...")
            # Use a smaller model and disable safetensors
            self.pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float32,  # Use float32 for CPU
                safety_checker=None,
                use_safetensors=False,  # Disable safetensors
                local_files_only=False,
                cache_dir="model_cache"  # Specify a local cache directory
            )
            
            # Use DPM++ 2M scheduler
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )
            
            # Move to device
            self.pipe = self.pipe.to(self.device)
            
            # Enable memory efficient settings
            self.pipe.enable_attention_slicing()
            
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            raise

    def enhance_prompt(self, prompt: str) -> str:
        """
        Enhance the prompt with medical-specific details
        """
        medical_quality_terms = (
            "medical illustration, highly detailed, professional medical diagram, "
            "anatomically correct, medical textbook quality, clear lighting"
        )
        return f"{prompt}, {medical_quality_terms}"

    def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 20,  # Reduced steps for faster generation
        guidance_scale: float = 7.5,
        width: int = 512,  # Standard size
        height: int = 512,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate a biomedical image
        """
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
            
        if negative_prompt is None:
            negative_prompt = (
                "blurry, low quality, low resolution, cropped, text, watermark, "
                "signature, out of frame, deformed, ugly, bad anatomy"
            )

        enhanced_prompt = self.enhance_prompt(prompt)
        
        try:
            print(f"Generating image for prompt: {prompt}")
            with torch.inference_mode():
                result = self.pipe(
                    prompt=enhanced_prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    generator=generator
                )
                
            image = result.images[0]
            print("Image generated successfully!")
            
            return {
                "image": image,
                "parameters": {
                    "prompt": prompt,
                    "enhanced_prompt": enhanced_prompt,
                    "negative_prompt": negative_prompt,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "width": width,
                    "height": height,
                    "seed": seed
                }
            }
            
        except Exception as e:
            print(f"Error generating image: {str(e)}")
            return None

    def save_image(
        self,
        image: Image.Image,
        save_dir: str,
        filename: str,
        save_metadata: bool = True,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Save the generated image and optionally its metadata
        """
        try:
            os.makedirs(save_dir, exist_ok=True)
            
            # Save image
            image_path = os.path.join(save_dir, filename)
            image.save(image_path)
            print(f"Image saved to: {image_path}")
            
            # Save metadata if requested
            if save_metadata and metadata:
                metadata_path = os.path.join(save_dir, f"{filename}.json")
                import json
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                print(f"Metadata saved to: {metadata_path}")
            
            return image_path
        except Exception as e:
            print(f"Error saving image: {str(e)}")
            return None 