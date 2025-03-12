import pandas as pd
import imgaug.augmenters as iaa
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

# Load CSV file with annotations
csv_file_path = 'D:\\research\\colonoscopy\\ai_colon\\polyp_images\\keys\\all_keys_labmod.csv'
df = pd.read_csv(csv_file_path)

# Path to the folder containing your original images
image_folder = 'D:\\research\\colonoscopy\\ai_colon\\polyp_images\\all'

seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # horizontal flips
    iaa.Flipud(0.5),  # vertical flips
    iaa.Multiply((0.8, 1.2), per_channel=0.2),  # random brightness
    iaa.GaussianBlur(sigma=(0.0, 1.0)),  # random Gaussian blur
    iaa.AdditiveGaussianNoise(scale=(0, 0.02*255)),  # random Gaussian noise
    iaa.LinearContrast((0.5, 1.5)),  # contrast normalization
    # Add more augmentations based on your needs
])


# Initialize an empty list to store augmented rows
augmented_rows = []

# Augment images and update bounding boxes in the DataFrame
for index, row in tqdm(df.iterrows(), total=len(df), desc="Augmenting images"):
    image_name = row['image_name']  # Replace with the column name in your CSV containing image names
    image_path = os.path.join(image_folder, image_name)

    # Convert PIL Image to NumPy array
    image = np.array(Image.open(image_path))

    # Extract bounding box coordinates
    x = row['x']
    y = row['y']
    width = row['width']
    height = row['height']

    bounding_box = BoundingBox(x1=x, y1=y, x2=x + width, y2=y + height)
    bounding_boxes_on_image = BoundingBoxesOnImage([bounding_box], shape=image.shape)

    # Initialize an empty list to store augmented rows for the current image
    augmented_rows_single_image = []

    # Apply augmentation to the image and bounding box multiple times
    for i in range(4):  # Increase images 5 times
        augmented_images, augmented_bbs = seq(images=[image], bounding_boxes=[bounding_boxes_on_image])

        # Convert the augmented image back to PIL Image
        augmented_image = Image.fromarray(augmented_images[0].astype('uint8'))

        # Save augmented image in the original image folder with a new name
        augmented_image_name = f"augmented_{image_name.rsplit('.', 1)[0]}_{i}.{image_name.rsplit('.', 1)[1]}"
        augmented_image_path = os.path.join(image_folder, augmented_image_name)
        augmented_image.save(augmented_image_path)

        # Update augmented rows with new bounding box coordinates and features
        augmented_row = row.copy()
        augmented_row['x'] = augmented_bbs[0].bounding_boxes[0].x1
        augmented_row['y'] = augmented_bbs[0].bounding_boxes[0].y1
        augmented_row['width'] = augmented_bbs[0].bounding_boxes[0].x2 - augmented_bbs[0].bounding_boxes[0].x1
        augmented_row['height'] = augmented_bbs[0].bounding_boxes[0].y2 - augmented_bbs[0].bounding_boxes[0].y1
        augmented_row['image_name'] = augmented_image_name  # Use the new image name
        augmented_rows_single_image.append(augmented_row)

    # Concatenate the augmented rows for the current image
    augmented_rows.extend(augmented_rows_single_image)

# Concatenate the original DataFrame with all augmented rows
augmented_df = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)

# Save the updated CSV file
updated_csv_path = 'D:\\research\\colonoscopy\\ai_colon\\polyp_images\\keys\\all_keys_labmod.csv'
augmented_df.to_csv(updated_csv_path, index=False)
