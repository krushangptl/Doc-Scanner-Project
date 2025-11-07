from PIL import Image
import numpy as np

# Load the original image
input_path = (
    "/home/krushang/radhey/personal/college/Doc-Scanner-Project/test/printed_guj.png"
)
image = Image.open(input_path).convert("RGB")

# Convert image to NumPy array
img_array = np.array(image, dtype=np.float32)

# Generate random noise (Gaussian noise)
noise = np.random.normal(
    0, 25, img_array.shape
)  # mean=0, std=25 (adjust for more/less noise)

# Add noise to image and clip values to valid range [0, 255]
noisy_img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

# Convert back to PIL Image
noisy_image = Image.fromarray(noisy_img_array)

# Save the noisy image
noisy_image.save(
    "/home/krushang/radhey/personal/college/Doc-Scanner-Project/test/noise_guj.png"
)
print("Noisy Image Created")
