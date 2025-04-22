from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
from typing import Tuple, Optional
from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler
)
import torch
import cv2

class ImageEditor:
    @staticmethod
    def adjust_brightness(image: Image.Image, factor: float) -> Image.Image:
        """Adjust image brightness"""
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    @staticmethod
    def adjust_contrast(image: Image.Image, factor: float) -> Image.Image:
        """Adjust image contrast"""
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    @staticmethod
    def adjust_sharpness(image: Image.Image, factor: float) -> Image.Image:
        """Adjust image sharpness"""
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(factor)

    @staticmethod
    def adjust_color(image: Image.Image, factor: float) -> Image.Image:
        """Adjust image color saturation"""
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)

    @staticmethod
    def apply_gaussian_blur(image: Image.Image, radius: float) -> Image.Image:
        """Apply Gaussian blur"""
        return image.filter(ImageFilter.GaussianBlur(radius))

    @staticmethod
    def apply_unsharp_mask(
        image: Image.Image,
        radius: float = 2,
        percent: int = 150,
        threshold: int = 3
    ) -> Image.Image:
        """Apply unsharp mask for detail enhancement"""
        return image.filter(ImageFilter.UnsharpMask(radius, percent, threshold))

    @staticmethod
    def adjust_gamma(image: Image.Image, gamma: float) -> Image.Image:
        """Adjust image gamma"""
        return ImageOps.autocontrast(image.point(lambda x: ((x / 255) ** gamma) * 255))

    @staticmethod
    def crop_image(
        image: Image.Image,
        box: Tuple[int, int, int, int]  # (left, top, right, bottom)
    ) -> Image.Image:
        """Crop image to specified box"""
        return image.crop(box)

    @staticmethod
    def resize_image(
        image: Image.Image,
        size: Tuple[int, int],
        resample: int = Image.LANCZOS
    ) -> Image.Image:
        """Resize image while maintaining aspect ratio"""
        return image.resize(size, resample=resample)

    @staticmethod
    def add_border(
        image: Image.Image,
        border: int,
        color: Tuple[int, int, int] = (255, 255, 255)
    ) -> Image.Image:
        """Add border to image"""
        return ImageOps.expand(image, border=border, fill=color)

    def apply_medical_enhancement(
        self,
        image: Image.Image,
        detail_enhancement: float = 1.2,
        contrast_boost: float = 1.1,
        sharpness: float = 1.3
    ) -> Image.Image:
        """
        Apply a combination of enhancements suitable for medical images
        """
        # Enhance details
        image = self.apply_unsharp_mask(image)
        
        # Boost contrast slightly
        image = self.adjust_contrast(image, contrast_boost)
        
        # Increase sharpness
        image = self.adjust_sharpness(image, sharpness)
        
        return image

    def optimize_for_print(
        self,
        image: Image.Image,
        dpi: int = 300,
        border_size: int = 50
    ) -> Image.Image:
        """
        Optimize image for printing (e.g., for medical publications)
        """
        # Add white border
        image = self.add_border(image, border_size, (255, 255, 255))
        
        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Set DPI
        image.info['dpi'] = (dpi, dpi)
        
        return image

class AdvancedImageEditor:
    def __init__(self):
        """Initialize the advanced image editor with InstructPix2Pix and ControlNet"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize InstructPix2Pix
        print("Loading InstructPix2Pix model...")
        self.pix2pix = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None
        ).to(self.device)
        
        # Initialize ControlNet for edge detection
        print("Loading ControlNet model...")
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        # Initialize ControlNet pipeline
        self.controlnet_pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None
        ).to(self.device)
        
        # Use better scheduler
        self.controlnet_pipe.scheduler = UniPCMultistepScheduler.from_config(
            self.controlnet_pipe.scheduler.config
        )
        
        # Enable memory efficient settings
        self.pix2pix.enable_attention_slicing()
        self.controlnet_pipe.enable_attention_slicing()

    def edit_with_instruction(
        self,
        image: Image.Image,
        instruction: str,
        num_inference_steps: int = 20,
        image_guidance_scale: float = 1.0,
        guidance_scale: float = 7.5
    ) -> Image.Image:
        """
        Edit image using InstructPix2Pix based on text instruction
        Args:
            image: Source image
            instruction: Text instruction for editing
            num_inference_steps: Number of denoising steps
            image_guidance_scale: How much to attend to the input image
            guidance_scale: How much to attend to the text instruction
        """
        try:
            # Ensure image is in RGB mode
            image = image.convert("RGB")
            
            # Resize image if needed
            if max(image.size) > 768:
                image.thumbnail((768, 768), Image.LANCZOS)
            
            # Generate edited image
            edited_image = self.pix2pix(
                instruction,
                image=image,
                num_inference_steps=num_inference_steps,
                image_guidance_scale=image_guidance_scale,
                guidance_scale=guidance_scale
            ).images[0]
            
            return edited_image
            
        except Exception as e:
            print(f"Error in edit_with_instruction: {str(e)}")
            return image

    def edit_with_controlnet(
        self,
        image: Image.Image,
        prompt: str,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0
    ) -> Image.Image:
        """
        Edit image using ControlNet with edge detection
        Args:
            image: Source image
            prompt: Text prompt for the desired modification
            num_inference_steps: Number of denoising steps
            guidance_scale: How much to attend to the text prompt
            controlnet_conditioning_scale: How much to attend to the control signal
        """
        try:
            # Convert to RGB
            image = image.convert("RGB")
            
            # Resize if needed
            if max(image.size) > 768:
                image.thumbnail((768, 768), Image.LANCZOS)
            
            # Convert to numpy array for edge detection
            image_np = np.array(image)
            
            # Detect edges using Canny
            low_threshold = 100
            high_threshold = 200
            edges = cv2.Canny(image_np, low_threshold, high_threshold)
            edges = edges[:, :, None]
            edges = np.concatenate([edges, edges, edges], axis=2)
            edge_image = Image.fromarray(edges)
            
            # Generate image using ControlNet
            edited_image = self.controlnet_pipe(
                prompt,
                image=edge_image,
                control_image=edge_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_conditioning_scale
            ).images[0]
            
            return edited_image
            
        except Exception as e:
            print(f"Error in edit_with_controlnet: {str(e)}")
            return image

    def image_to_image(
        self,
        image: Image.Image,
        prompt: str,
        strength: float = 0.8,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5
    ) -> Image.Image:
        """
        Perform image-to-image translation
        Args:
            image: Source image
            prompt: Text prompt for the desired modification
            strength: How much to transform the image (0-1)
            num_inference_steps: Number of denoising steps
            guidance_scale: How much to attend to the text prompt
        """
        try:
            # Use ControlNet pipeline for image-to-image
            return self.edit_with_controlnet(
                image,
                prompt,
                num_inference_steps,
                guidance_scale,
                strength
            )
        except Exception as e:
            print(f"Error in image_to_image: {str(e)}")
            return image 