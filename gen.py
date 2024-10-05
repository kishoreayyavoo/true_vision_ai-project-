import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Function to read attributes from a text file
def read_attributes(file_path):
    attributes = {}
    with open(file_path, 'r') as f:
        for line in f:
            key, value = line.strip().split(': ')
            attributes[key] = value
    return attributes

# Load the StyleGAN2 model
class StyleGAN2:
    def __init__(self, model_path):
        # Load the pre-trained StyleGAN2 model
        self.model = torch.load(model_path)
        self.model.eval()

    def generate(self, z):
        with torch.no_grad():
            img = self.model(z)
        return img

# Generate a random latent vector (512-dimensional)
def generate_latent_vector(size):
    return torch.randn(1, size)

# Main function to generate a face
def main():
    # Read attributes from the text file
    attributes_file = "D:/deepfake_detection_project/outputhair/detection_results.txt"  # Update this path to your attributes file
    attributes = read_attributes(attributes_file)
    
    # For now, we just print the attributes to see what we have
    print("Attributes:", attributes)

    model_path = "stylegan2-ffhq-config-f.pt"  # Update this path to your model file
    stylegan = StyleGAN2(model_path)

    # Generate a latent vector
    latent_vector = generate_latent_vector(512)  # StyleGAN2 typically uses a latent size of 512

    # Generate the image
    generated_image = stylegan.generate(latent_vector)

    # Save and display the image
    img = generated_image.squeeze().permute(1, 2, 0)  # Change the tensor shape from (C, H, W) to (H, W, C)
    img = (img + 1) / 2  # Normalize to [0, 1]
    img = Image.fromarray((img.numpy() * 255).astype(np.uint8))

    img.save("generated_face.png")
    plt.imshow(img)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
