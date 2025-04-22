from src.image_generation.img_generator import BiomedicalImageGenerator
import os

def test_image_generation():
    try:
        # Create model cache directory
        os.makedirs("model_cache", exist_ok=True)
        
        # Initialize generator
        print("Initializing image generator...")
        generator = BiomedicalImageGenerator()
        
        # Test prompt
        prompt = "Simple 2D medical diagram of a cell"  # Simple test prompt
        
        # Generate image
        result = generator.generate_image(
            prompt=prompt,
            num_inference_steps=20,
            guidance_scale=7.5,
            width=512,
            height=512,
            seed=42
        )
        
        if result:
            # Create output directory
            os.makedirs("generated_images", exist_ok=True)
            
            # Save original image
            original_path = generator.save_image(
                result["image"],
                "generated_images",
                "test_image.png",
                save_metadata=True,
                metadata=result["parameters"]
            )
            
            if original_path:
                print(f"Test completed successfully!")
            else:
                print("Failed to save the generated image")
        else:
            print("Failed to generate image")
            
    except Exception as e:
        print(f"Error during image generation test: {str(e)}")
        raise

if __name__ == "__main__":
    test_image_generation() 