import os
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image, ImageDraw

def draw_head_bbox(pil_img, bbox):
    """
    Draw the bounding box (head_bbox) onto the given PIL image.
    bbox is [ymin, xmin, ymax, xmax] in normalized [0,1].
    """
    if bbox is None or len(bbox) != 4:
        return pil_img
    w, h = pil_img.size
    ymin, xmin, ymax, xmax = bbox  # normalized coordinates
    left, right = xmin * w, xmax * w
    top, bottom = ymin * h, ymax * h
    draw = ImageDraw.Draw(pil_img)
    draw.rectangle([left, top, right, bottom], outline="red", width=3)
    return pil_img

def prettify_breed_name(breed_label_str: str) -> str:
    """
    Convert a label like 'american_bulldog' into 'American_Bulldog'.
    """
    parts = breed_label_str.split("_")
    parts = [p.capitalize() for p in parts]
    return "_".join(parts)

def main():
    # Where to store final images
    data_root = "../Data"
    os.makedirs(data_root, exist_ok=True)
    
    # Lists to collect image paths for later
    cats_no_box_paths = []
    cats_with_box_lines = []
    dogs_no_box_paths = []
    dogs_with_box_lines = []
    
    # Download the dataset (using both train and test splits)
    ds, info = tfds.load(
        "oxford_iiit_pet",
        split="train+test",
        with_info=True,
        as_supervised=False
    )
    
    # Label names (e.g., "Abyssinian", "american_bulldog", …)
    label_names = info.features["label"].names
    
    final_size = 224  # Final image dimensions
    
    count = 0
    for example in ds:
        image_tensor = example["image"]            # shape [H, W, 3]
        label = example["label"].numpy()             # integer index into label_names
        species = example["species"].numpy()         # 0 for cats, 1 for dogs
        bbox = example["head_bbox"].numpy()          # [4] normalized coordinates: [ymin, xmin, ymax, xmax]
        file_name = example["file_name"].numpy().decode("utf-8")  # e.g., "Abyssinian_1.jpg"
        
        # Determine species folder name
        species_str = "cats" if species == 0 else "dogs"
        
        # Determine breed folder name from label_names
        raw_breed_str = label_names[label]
        breed_folder = prettify_breed_name(raw_breed_str)
        
        # (A) Save "no_box" version
        no_box_dir = os.path.join(data_root, "no_box", species_str, breed_folder)
        os.makedirs(no_box_dir, exist_ok=True)
        pil_img_no_box = Image.fromarray(image_tensor.numpy())
        pil_img_no_box = pil_img_no_box.resize((final_size, final_size), Image.Resampling.LANCZOS)
        out_no_box_path = os.path.join(no_box_dir, file_name)
        pil_img_no_box.save(out_no_box_path)
        if species == 0:
            cats_no_box_paths.append(out_no_box_path)
        else:
            dogs_no_box_paths.append(out_no_box_path)
        
        # (B) Save "with_box" version
        with_box_dir = os.path.join(data_root, "with_box", species_str, breed_folder)
        os.makedirs(with_box_dir, exist_ok=True)
        pil_img_with_box = Image.fromarray(image_tensor.numpy())
        pil_img_with_box = draw_head_bbox(pil_img_with_box, bbox)
        # Resize after drawing the bounding box
        pil_img_with_box = pil_img_with_box.resize((final_size, final_size), Image.Resampling.LANCZOS)
        out_with_box_path = os.path.join(with_box_dir, file_name)
        pil_img_with_box.save(out_with_box_path)
        
        # Compute bounding box coordinates for the final 224x224 image.
        # Since bbox is normalized, we can directly scale by final_size.
        if bbox is not None and len(bbox) == 4:
            ymin, xmin, ymax, xmax = bbox
            x1_final = int(round(xmin * final_size))
            y1_final = int(round(ymin * final_size))
            x2_final = int(round(xmax * final_size))
            y2_final = int(round(ymax * final_size))
            bbox_coords_str = f"{x1_final} {y1_final} {x2_final} {y2_final}"
        else:
            bbox_coords_str = "0 0 0 0"
        
        # Create a line in the desired format: "/path/to/image.jpg x1 y1 x2 y2"
        with_box_line = f"{out_with_box_path} {bbox_coords_str}"
        
        if species == 0:
            cats_with_box_lines.append(with_box_line)
        else:
            dogs_with_box_lines.append(with_box_line)
        
        count += 1
        if count % 100 == 0:
            print(f"Processed {count} images...")
    
    # Write out separate text files for cats and dogs (no_box and with_box)
    with open("../Data/paths_cats_no_box.txt", "w") as f:
        for p in cats_no_box_paths:
            f.write(p + "\n")
    with open("../Data/paths_cats_with_box.txt", "w") as f:
        for line in cats_with_box_lines:
            f.write(line + "\n")
    with open("../Data/paths_dogs_no_box.txt", "w") as f:
        for p in dogs_no_box_paths:
            f.write(p + "\n")
    with open("../Data/paths_dogs_with_box.txt", "w") as f:
        for line in dogs_with_box_lines:
            f.write(line + "\n")
    
    print(f"Done! Processed a total of {count} images.")

if __name__ == "__main__":
    main()
