import os
from PIL import Image
import numpy as np

def parse_path_info(path):
    """
    Given a file path like "../Data/no_box/cats/Abyssinian/Abyssinian_1.jpg",
    returns:
      - species: 0 for cats, 1 for dogs
      - breed: the breed name (e.g. "Abyssinian")
    """
    parts = os.path.normpath(path).split(os.sep)
    species = None
    breed = None
    if "cats" in parts:
        species = 0
        idx = parts.index("cats")
        if idx + 1 < len(parts):
            breed = parts[idx + 1]
    elif "dogs" in parts:
        species = 1
        idx = parts.index("dogs")
        if idx + 1 < len(parts):
            breed = parts[idx + 1]
    return species, breed

def read_paths(txt_file):
    """
    Reads a text file where each line is an image path.
    Returns a list of tuples: (path, species, breed)
    """
    data = []
    if not os.path.exists(txt_file):
        raise FileNotFoundError(f"Cannot find {txt_file}")
    with open(txt_file, 'r') as f:
        for line in f:
            path = line.strip()
            if path:
                species, breed = parse_path_info(path)
                data.append((path, species, breed))
    return data

def load_and_normalize_image(path):
    with Image.open(path).convert("RGB") as img:
        arr = np.array(img).astype(np.float32) / 255.0
    return arr

def load_entire_data(box_type="no_box", shuffle_data=True):
    """
    Loads all images from the specified box_type along with labels.
    
    Returns:
      images_np: NumPy array of shape (N, 224, 224, 3)
      species_labels_np: NumPy array of shape (N,) with 0 for cats, 1 for dogs
      breed_labels_np: NumPy array of shape (N,) with integer labels for each breed
    """
    if box_type == "no_box":
        cats_txt = "../Data/paths_cats_no_box.txt"
        dogs_txt = "../Data/paths_dogs_no_box.txt"
    elif box_type == "with_box":
        cats_txt = "../Data/paths_cats_with_box.txt"
        dogs_txt = "../Data/paths_dogs_with_box.txt"
    else:
        raise ValueError("box_type must be either 'no_box' or 'with_box'.")
    
    data_cats = read_paths(cats_txt)
    data_dogs = read_paths(dogs_txt)
    data = data_cats + data_dogs
    data.sort(key=lambda x: x[0])
    
    X = [item[0] for item in data]
    species_labels = [item[1] for item in data]
    breed_strings = [item[2] for item in data]
    
    # Build mapping from breed strings to integer labels
    unique_breeds = sorted(list(set(breed_strings)))
    breed_to_int = {breed: i for i, breed in enumerate(unique_breeds)}
    breed_labels = [breed_to_int[b] for b in breed_strings]
    
    imgs = [load_and_normalize_image(path) for path in X]
    images_np = np.stack(imgs, axis=0)
    species_labels_np = np.array(species_labels, dtype=np.int64)
    breed_labels_np = np.array(breed_labels, dtype=np.int64)
    
    if shuffle_data:
        indices = np.arange(len(images_np))
        np.random.shuffle(indices)
        images_np = images_np[indices]
        species_labels_np = species_labels_np[indices]
        breed_labels_np = breed_labels_np[indices]
    
    print(f"Loaded {len(images_np)} images (box_type={box_type}).")
    print("images_np.shape =", images_np.shape,
          "| species_labels_np.shape =", species_labels_np.shape,
          "| breed_labels_np.shape =", breed_labels_np.shape)
    print("Unique breeds:", unique_breeds)
    return images_np, species_labels_np, breed_labels_np
