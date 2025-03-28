import os
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np

def read_paths(txt_file, label):
    """
    Reads file paths from a text file and assigns a label (0=cats, 1=dogs).
    Returns a list of (filepath, label) tuples.
    """
    data = []
    if not os.path.exists(txt_file):
        raise FileNotFoundError(f"Cannot find {txt_file}")
    with open(txt_file, 'r') as f:
        for line in f:
            path = line.strip()
            if path:
                data.append((path, label))
    return data

def load_and_normalize_image(path):
    """
    Loads an image in RGB and returns a normalized np.array in [0..1].
    Shape: (height, width, 3)
    """
    with Image.open(path).convert("RGB") as img:
        arr = np.array(img).astype(np.float32) / 255.0
    return arr

def process_images(box_type="no_box"):
    """
    - box_type: "no_box" or "with_box"
    - Reads paths from the respective paths_*.txt files,
      stratifies into 70/15/15 train/val/test,
      normalizes all images, and prints resulting shapes/stats.
    """

    # 1. Decide which text files to read
    if box_type == "no_box":
        cats_txt = "../Data/paths_cats_no_box.txt"
        dogs_txt = "../Data/paths_dogs_no_box.txt"
    elif box_type == "with_box":
        cats_txt = "../Data/paths_cats_with_box.txt"
        dogs_txt = "../Data/paths_dogs_with_box.txt"
    else:
        raise ValueError("box_type must be either 'no_box' or 'with_box'.")

    # 2. Read file paths & assign cat/dog labels
    cats_data = read_paths(cats_txt, label=0)  # 0 = cat
    dogs_data = read_paths(dogs_txt, label=1)  # 1 = dog

    # Combine
    data = cats_data + dogs_data

    # Separate into X (paths) and y (labels)
    X = [item[0] for item in data]
    y = [item[1] for item in data]

    # 3. Stratified 70/15/15 split
    #    First split off 70% for train, then split the remaining 30% for val/test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.50,  # half of the 30% => 15% total
        random_state=42,
        stratify=y_temp
    )

    # 4. Load & normalize images
    train_imgs = [load_and_normalize_image(path) for path in X_train]
    val_imgs   = [load_and_normalize_image(path) for path in X_val]
    test_imgs  = [load_and_normalize_image(path) for path in X_test]

    # Convert lists of arrays into stacked arrays (N, H, W, 3)
    train_imgs_np = np.stack(train_imgs, axis=0)
    val_imgs_np   = np.stack(val_imgs, axis=0)
    test_imgs_np  = np.stack(test_imgs, axis=0)

    # 5. Calculate stats
    train_len = len(X_train)
    val_len   = len(X_val)
    test_len  = len(X_test)
    total_len = train_len + val_len + test_len

    # -- Percentage breakdown by class --
    cat_train = y_train.count(0)
    dog_train = y_train.count(1)
    cat_val   = y_val.count(0)
    dog_val   = y_val.count(1)
    cat_test  = y_test.count(0)
    dog_test  = y_test.count(1)

    cat_train_pct = 100.0 * cat_train / train_len if train_len > 0 else 0
    dog_train_pct = 100.0 * dog_train / train_len if train_len > 0 else 0
    cat_val_pct   = 100.0 * cat_val / val_len if val_len > 0 else 0
    dog_val_pct   = 100.0 * dog_val / val_len if val_len > 0 else 0
    cat_test_pct  = 100.0 * cat_test / test_len if test_len > 0 else 0
    dog_test_pct  = 100.0 * dog_test / test_len if test_len > 0 else 0

    # 6. Print stats
    print(f"=== Processing {box_type.upper()} ===")
    print("Train size:", train_len, " -> shape:", train_imgs_np.shape)
    print(f"  Cats: {cat_train_pct:.2f}%  Dogs: {dog_train_pct:.2f}%")
    print("Val size:  ", val_len,   " -> shape:", val_imgs_np.shape)
    print(f"  Cats: {cat_val_pct:.2f}%   Dogs: {dog_val_pct:.2f}%")
    print("Test size: ", test_len,  " -> shape:", test_imgs_np.shape)
    print(f"  Cats: {cat_test_pct:.2f}%  Dogs: {dog_test_pct:.2f}%")
    print("Total images:", total_len)

    # Return the Normalized Data splits
    return train_imgs_np, val_imgs_np, test_imgs_np

if __name__ == "__main__":
    # Just call the function here. Change to "with_box" if you like.
    train_data, val_data, test_data = process_images("no_box")
