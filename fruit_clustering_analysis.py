
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import base64
from PIL import Image
import io
import os
from sklearn.datasets import load_files
from pathlib import Path
import shutil

# This script simulates a file system to use sklearn's load_files
# It's a workaround for environments where we can't have actual file structures easily.

def setup_virtual_filesystem(base_path='virtual_fruits'):
    """Creates a temporary directory structure for fruit images."""
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    
    # Create directories for train and test sets for each fruit
    for fruit in ['apples', 'bananas', 'pineapples']:
        os.makedirs(os.path.join(base_path, 'train', fruit), exist_ok=True)
        os.makedirs(os.path.join(base_path, 'test', fruit), exist_ok=True)

    # For this simulation, we'll create dummy blank images
    # In a real scenario, you'd populate this with actual image files
    for fruit in ['apples', 'bananas', 'pineapples']:
        for i in range(10): # Create 10 dummy images per category
            img = Image.new('L', (100, 100), color = 'white' if fruit == 'apples' else 'gray' if fruit == 'bananas' else 'black')
            img.save(os.path.join(base_path, 'train', fruit, f'dummy_{i}.png'))


def main():
    VIRTUAL_FRUITS_DIR = 'virtual_fruits_data'
    try:
        # Load data from stdin
        payload = json.load(sys.stdin)
        user_image_data_url = payload.get('image')
        
        # Set up the dummy file system
        if not Path(VIRTUAL_FRUITS_DIR).exists():
             setup_virtual_filesystem(VIRTUAL_FRUITS_DIR)

        # Load files from the virtual directory
        fruits_train = load_files(os.path.join(VIRTUAL_FRUITS_DIR, 'train'))

        # Process the loaded images into a numpy array
        # This part simulates loading grayscale images
        images = []
        for filename in fruits_train.filenames:
            try:
                with Image.open(filename) as img:
                    # Convert to grayscale and resize
                    processed_img = img.convert('L').resize((100, 100))
                    images.append(np.array(processed_img))
            except Exception as e:
                # Skip corrupted or invalid images
                continue
        
        if not images:
            raise FileNotFoundError("Could not load any fruit images from the dataset.")

        fruits = np.array(images)

        # --- The logic from here is similar to the original, but adapted ---
        apple = fruits[fruits_train.target == 0].reshape(-1, 100*100)
        pineapple = fruits[fruits_train.target == 2].reshape(-1, 100*100)
        banana = fruits[fruits_train.target == 1].reshape(-1, 100*100)

        # Process user's image
        header, encoded = user_image_data_url.split(",", 1)
        image_data = base64.b64decode(encoded)
        user_image = Image.open(io.BytesIO(image_data)).convert('L').resize((100, 100))
        user_image_np = np.array(user_image)

        # Find which cluster the user image belongs to
        apple_mean = np.mean(apple, axis=0) if len(apple) > 0 else np.zeros(10000)
        pineapple_mean = np.mean(pineapple, axis=0) if len(pineapple) > 0 else np.zeros(10000)
        banana_mean = np.mean(banana, axis=0) if len(banana) > 0 else np.zeros(10000)

        dist_apple = np.mean(np.abs(user_image_np.flatten() - apple_mean))
        dist_pineapple = np.mean(np.abs(user_image_np.flatten() - pineapple_mean))
        dist_banana = np.mean(np.abs(user_image_np.flatten() - banana_mean))

        distances = {'apple': dist_apple, 'pineapple': dist_pineapple, 'banana': dist_banana}
        user_image_cluster = min(distances, key=distances.get)

        def find_closest(cluster_data_3d, mean_img_flat):
            if len(cluster_data_3d) == 0: return []
            abs_diff = np.abs(cluster_data_3d - mean_img_flat.reshape(100, 100))
            abs_mean = np.mean(abs_diff, axis=(1, 2))
            closest_indices = np.argsort(abs_mean)[:5]
            images_b64 = []
            for idx in closest_indices:
                img = Image.fromarray(cluster_data_3d[idx])
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                images_b64.append('data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('utf-8'))
            return images_b64

        closest_images = {
            'apple': find_closest(fruits[fruits_train.target == 0], apple_mean),
            'banana': find_closest(fruits[fruits_train.target == 1], banana_mean),
            'pineapple': find_closest(fruits[fruits_train.target == 2], pineapple_mean)
        }
        
        def generate_mean_image(mean_flat):
            mean_img = mean_flat.reshape(100,100)
            img = Image.fromarray(mean_img.astype(np.uint8))
            buf = io.BytesIO()
            img.save(buf, format='PNG')
            return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode('utf-8')
            
        mean_images = {
            'apple': generate_mean_image(apple_mean),
            'banana': generate_mean_image(banana_mean),
            'pineapple': generate_mean_image(pineapple_mean),
        }

        response = {
            'user_image_cluster': user_image_cluster,
            'closest_images': closest_images,
            'mean_images': mean_images,
            'histograms': {}
        }
        
        print(json.dumps(response))

    except FileNotFoundError:
        print(json.dumps({"error": "The required fruit dataset (fruits_300.zip) was not found on the server."}), file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)
    finally:
        # Clean up virtual file system
        if os.path.exists(VIRTUAL_FRUITS_DIR):
            shutil.rmtree(VIRTUAL_FRUITS_DIR)


if __name__ == '__main__':
    main()
