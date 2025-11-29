"""
Generate captions for SuperTuxKart dataset for CLIP training.
"""

import json
import random
from pathlib import Path
import fire
import matplotlib.pyplot as plt

from .generate_qa import draw_detections, extract_kart_objects, extract_track_info


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate captions for a specific view.
    
    Returns:
        List of caption strings describing the scene
    """
    # Extract kart objects and track info (reuse from generate_qa)
    kart_objects = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)
    
    captions = []
    
    # Find the ego car
    ego_car = None
    for kart in kart_objects:
        if kart["is_center_kart"]:
            ego_car = kart
            break
    
    if ego_car is None:
        return []
    
    ego_center_x, ego_center_y = ego_car["center"]
    
    # Generate multiple simple captions instead of one complex one
    # 1. Ego car caption
    captions.append(f"{ego_car['kart_name']} is the ego car.")
    
    # 2. Track name caption
    captions.append(f"The track is {track_name}.")
    
    # 3. Counting caption
    captions.append(f"There are {len(kart_objects)} karts in the scene.")
    
    # 4. Add some positional captions (but not all - keep it manageable)
    # Only add captions for up to 2-3 other karts to avoid overwhelming the model
    other_karts = [k for k in kart_objects if not k["is_center_kart"]]
    for kart in other_karts[:3]:  # Limit to first 3 other karts
        kart_center_x, kart_center_y = kart["center"]
        kart_name = kart["kart_name"]
        
        # Determine left/right
        lr = "left" if kart_center_x < ego_center_x else "right"
        
        # Determine front/back
        fb = "front" if kart_center_y < ego_center_y else "back"
        
        # Simpler caption format
        captions.append(f"{kart_name} is to the {lr}.")
        captions.append(f"{kart_name} is in {fb}.")
    
    return captions


def check_caption(info_file: str, view_index: int):
    """Check captions for a specific info file and view index."""
    captions = generate_caption(info_file, view_index)

    print("\nCaptions:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"View {view_index}")
    plt.show()


def generate_all_captions(split='train'):
    """
    Generate caption pairs for all images in the dataset.
    Each image gets ONE simple caption (not all concatenated).
    """
    data_dir = Path(f'data/{split}')
    all_caption_pairs = []
    
    # Process all images
    image_files = sorted(data_dir.glob('*_*_im.jpg'))
    print(f"Found {len(image_files)} images in {data_dir}")
    
    for image_path in image_files:
        # Get view index from filename
        _, view_index_str, _ = image_path.stem.split('_')
        view_index = int(view_index_str)
        
        # Get corresponding info file
        frame_id = image_path.stem.split('_')[0]
        info_path = data_dir / f"{frame_id}_info.json"
        
        if not info_path.exists():
            continue
        
        # Generate captions for this image
        captions = generate_caption(str(info_path), view_index)
        
        if not captions:
            continue
        
        # Get relative image path
        relative_image_path = str(image_path.relative_to(data_dir.parent))
        
        # KEY CHANGE: Create multiple training pairs instead of concatenating
        # Each caption becomes a separate training example
        for caption in captions:
            all_caption_pairs.append({
                "caption": caption,  # Individual caption, not combined!
                "image_file": relative_image_path
            })
    
    print(f"Generated {len(all_caption_pairs)} caption pairs")
    
    # Save to JSON
    output_file = data_dir / f'{split}_captions.json'
    with open(output_file, 'w') as f:
        json.dump(all_caption_pairs, f, indent=2)
    
    print(f"Saved to {output_file}")
    return all_caption_pairs


def main():
    fire.Fire({
        "check": check_caption,
        "generate": generate_all_captions
    })


if __name__ == "__main__":
    main()