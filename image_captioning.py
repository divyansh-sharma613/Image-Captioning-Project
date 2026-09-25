from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

print("Loading Image Captioning Model...")

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

print("Model loaded successfully.")

image_path = input("Enter image path: ")

image = Image.open(image_path).convert("RGB")

inputs = processor(
    images=image,
    return_tensors="pt"
)

output = model.generate(
    **inputs,
    max_new_tokens=30
)

caption = processor.decode(
    output[0],
    skip_special_tokens=True
)

print()
print("Generated Caption:")
print(caption)