"""
Generate question-answer pairs for SuperTuxKart dataset.
"""

import json
import random
from pathlib import Path
from collections import defaultdict


def extract_kart_objects(
    info_path: str, view_index: int, img_width: int = 150, img_height: int = 100, min_box_size: int = 5
) -> list:
    """
    Extract kart objects from the info.json file, including their center points and identify the center kart.
    Filters out karts that are out of sight (outside the image boundaries).

    Args:
        info_path: Path to the corresponding info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of kart objects, each containing:
        - instance_id: The track ID of the kart
        - kart_name: The name of the kart
        - center: (x, y) coordinates of the kart's center
        - is_center_kart: Boolean indicating if this is the kart closest to image center
    """
    ORIGINAL_WIDTH = 600
    ORIGINAL_HEIGHT = 400

    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)
    
    # Get kart names
    kart_names = info["karts"]
    
    # Get detections for this view
    if view_index >= len(info["detections"]):
        return []
    
    frame_detections = info["detections"][view_index]
    
    # Calculate scaling factors (from 600x400 to 150x100)
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT
    
    kart_objects = []
    
    # Process each detection
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)
        
        # Only process karts (class_id == 1)
        if class_id != 1:
            continue
        
        # Scale coordinates to image size
        x1_scaled = x1 * scale_x
        y1_scaled = y1 * scale_y
        x2_scaled = x2 * scale_x
        y2_scaled = y2 * scale_y
        
        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue
        
        # Filter out karts that are completely out of bounds
        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue
        
        # Calculate center point of the bounding box
        center_x = (x1_scaled + x2_scaled) / 2
        center_y = (y1_scaled + y2_scaled) / 2
        
        # Get kart name using track_id as index
        kart_name = kart_names[track_id]
        
        kart_objects.append({
            "instance_id": track_id,
            "kart_name": kart_name,
            "center": (center_x, center_y),
            "is_center_kart": False  # Will update this next
        })
    
    # Identify the ego car (kart closest to image center)
    if kart_objects:
        image_center_x = img_width / 2
        image_center_y = img_height / 2
        
        min_distance = float('inf')
        center_kart_idx = 0
        
        for idx, kart in enumerate(kart_objects):
            cx, cy = kart["center"]
            # Calculate Euclidean distance to image center
            distance = ((cx - image_center_x) ** 2 + (cy - image_center_y) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                center_kart_idx = idx
        
        # Mark the center kart as ego car
        kart_objects[center_kart_idx]["is_center_kart"] = True
    
    return kart_objects


def extract_track_info(info_path: str) -> str:
    """
    Extract track information from the info.json file.

    Args:
        info_path: Path to the info.json file

    Returns:
        Track name as a string
    """
    with open(info_path) as f:
        info = json.load(f)
    
    return info["track"]


def generate_qa_pairs(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate question-answer pairs for a given view.

    Args:
        info_path: Path to the info.json file
        view_index: Index of the view to analyze
        img_width: Width of the image (default: 150)
        img_height: Height of the image (default: 100)

    Returns:
        List of dictionaries, each containing a question and answer
    """
    # Extract kart objects and track info
    kart_objects = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)
    
    qa_pairs = []
    
    # Find the ego car (center kart)
    ego_car = None
    for kart in kart_objects:
        if kart["is_center_kart"]:
            ego_car = kart
            break
    
    # If no ego car found, return empty list
    if ego_car is None:
        return []
    
    ego_center_x, ego_center_y = ego_car["center"]
    
    # 1. Ego car question
    qa_pairs.append({
        "question": "What kart is the ego car?",
        "answer": ego_car["kart_name"]
    })
    
    # 2. Total karts question
    qa_pairs.append({
        "question": "How many karts are there in the scenario?",
        "answer": str(len(kart_objects))
    })
    
    # 3. Track information question
    qa_pairs.append({
        "question": "What track is this?",
        "answer": track_name
    })
    
    # 4. Relative position questions for each non-ego kart
    for kart in kart_objects:
        if kart["is_center_kart"]:
            continue
        
        kart_center_x, kart_center_y = kart["center"]
        kart_name = kart["kart_name"]
        
        # Determine left/right position
        if kart_center_x < ego_center_x:
            lr_position = "left"
        else:
            lr_position = "right"
        
        qa_pairs.append({
            "question": f"Is {kart_name} to the left or right of the ego car?",
            "answer": lr_position
        })
        
        # Determine front/behind position
        # CORRECTED: Y=0 is TOP, Y increases DOWNWARD
        # Lower Y = higher in image = in front
        # Higher Y = lower in image = behind
        if kart_center_y < ego_center_y:
            fb_position = "front"  # ✅ FIXED
        else:
            fb_position = "back"    # ✅ FIXED
        
        qa_pairs.append({
            "question": f"Is {kart_name} in front of or behind the ego car?",
            "answer": fb_position
        })
        
        # Combined position question
        combined_position = f"{fb_position} and {lr_position}"
        qa_pairs.append({
            "question": f"Where is {kart_name} relative to the ego car?",
            "answer": combined_position
        })
    
    # 5. Counting questions - CORRECTED
    count_left = sum(1 for kart in kart_objects 
                     if not kart["is_center_kart"] and kart["center"][0] < ego_center_x)
    count_right = sum(1 for kart in kart_objects 
                      if not kart["is_center_kart"] and kart["center"][0] > ego_center_x)
    count_front = sum(1 for kart in kart_objects 
                      if not kart["is_center_kart"] and kart["center"][1] < ego_center_y)  # ✅ FIXED
    count_behind = sum(1 for kart in kart_objects 
                       if not kart["is_center_kart"] and kart["center"][1] > ego_center_y)  # ✅ FIXED
    
    qa_pairs.append({
        "question": "How many karts are to the left of the ego car?",
        "answer": str(count_left)
    })
    
    qa_pairs.append({
        "question": "How many karts are to the right of the ego car?",
        "answer": str(count_right)
    })
    
    qa_pairs.append({
        "question": "How many karts are in front of the ego car?",
        "answer": str(count_front)
    })
    
    qa_pairs.append({
        "question": "How many karts are behind the ego car?",
        "answer": str(count_behind)
    })
    
    return qa_pairs


def balance_qa_pairs(all_qa_pairs, max_per_type=None):
    """
    Balance QA pairs by question type.
    
    Args:
        all_qa_pairs: List of all QA pairs
        max_per_type: Maximum number of QA pairs per question type (None for no limit)
    
    Returns:
        list: Balanced list of QA pairs
    """
    # Group by question type
    qa_by_type = defaultdict(list)
    for qa in all_qa_pairs:
        # Extract question type (first few words)
        question = qa['question']
        if 'What kart is the ego car?' in question:
            q_type = 'ego_car'
        elif 'How many karts are there' in question:
            q_type = 'total_karts'
        elif 'What track is this?' in question:
            q_type = 'track'
        elif 'to the left or right' in question:
            q_type = 'left_right'
        elif 'in front of or behind' in question:
            q_type = 'front_back'
        elif 'Where is' in question and 'relative to' in question:
            q_type = 'combined_position'
        elif 'How many karts are in front' in question:
            q_type = 'count_front'
        elif 'How many karts are behind' in question:
            q_type = 'count_behind'
        elif 'How many karts are to the left' in question:
            q_type = 'count_left'
        elif 'How many karts are to the right' in question:
            q_type = 'count_right'
        else:
            q_type = 'other'
        
        qa_by_type[q_type].append(qa)
    
    # Balance by sampling
    balanced_pairs = []
    for q_type, pairs in qa_by_type.items():
        if max_per_type and len(pairs) > max_per_type:
            pairs = random.sample(pairs, max_per_type)
        balanced_pairs.extend(pairs)
    
    random.shuffle(balanced_pairs)
    return balanced_pairs


def main():
    """Generate QA pairs for all images in the dataset."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate QA pairs for SuperTuxKart dataset')
    parser.add_argument('--data_dir', type=str, default='data/train',
                      help='Directory containing the dataset')
    parser.add_argument('--output', type=str, default='data/train/balanced_qa_pairs.json',
                      help='Output JSON file path')
    parser.add_argument('--max_per_type', type=int, default=None,
                      help='Maximum QA pairs per question type (None for no limit)')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    base_data_dir = data_dir.parent  # Get "data" from "data/train"
    all_qa_pairs = []
    
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
        
        # Generate QA pairs for this image
        qa_pairs = generate_qa_pairs(str(info_path), view_index)
        
        # Add the image_file path to each QA pair (relative to base data dir)
        relative_image_path = str(image_path.relative_to(base_data_dir))
        for qa in qa_pairs:
            qa["image_file"] = relative_image_path
        
        all_qa_pairs.extend(qa_pairs)
    
    print(f"Generated {len(all_qa_pairs)} QA pairs before balancing")
    
    # Balance QA pairs
    balanced_pairs = balance_qa_pairs(all_qa_pairs, max_per_type=args.max_per_type)
    print(f"Balanced to {len(balanced_pairs)} QA pairs")
    
    # Save to JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(balanced_pairs, f, indent=2)
    
    print(f"Saved QA pairs to {output_path}")

def draw_detections(
    image_path: str, info_path: str, font_scale: float = 0.5, thickness: int = 1, min_box_size: int = 5
):
    """Draw detection bounding boxes and labels on the image."""
    from PIL import Image, ImageDraw
    import numpy as np
    
    ORIGINAL_WIDTH = 600
    ORIGINAL_HEIGHT = 400
    
    # Read the image using PIL
    pil_image = Image.open(image_path)
    img_width, img_height = pil_image.size
    
    # Create a drawing context
    draw = ImageDraw.Draw(pil_image)
    
    # Read the info.json file
    with open(info_path) as f:
        info = json.load(f)
    
    # Extract frame ID and view index from image filename
    filename = Path(image_path).name
    parts = filename.split("_")
    view_index = int(parts[1])
    
    # Get the correct detection frame based on view index
    if view_index < len(info["detections"]):
        frame_detections = info["detections"][view_index]
    else:
        print(f"Warning: View index {view_index} out of range for detections")
        return np.array(pil_image)
    
    # Calculate scaling factors
    scale_x = img_width / ORIGINAL_WIDTH
    scale_y = img_height / ORIGINAL_HEIGHT
    
    # Draw each detection
    for detection in frame_detections:
        class_id, track_id, x1, y1, x2, y2 = detection
        class_id = int(class_id)
        track_id = int(track_id)
        
        if class_id != 1:
            continue
        
        # Scale coordinates to fit the current image size
        x1_scaled = int(x1 * scale_x)
        y1_scaled = int(y1 * scale_y)
        x2_scaled = int(x2 * scale_x)
        y2_scaled = int(y2 * scale_y)
        
        # Skip if bounding box is too small
        if (x2_scaled - x1_scaled) < min_box_size or (y2_scaled - y1_scaled) < min_box_size:
            continue
        
        if x2_scaled < 0 or x1_scaled > img_width or y2_scaled < 0 or y1_scaled > img_height:
            continue
        
        # Get color - ego car (track_id 0) in red, others in green
        if track_id == 0:
            color = (255, 0, 0)  # Red for ego
        else:
            color = (0, 255, 0)  # Green for others
        
        # Draw bounding box
        draw.rectangle([(x1_scaled, y1_scaled), (x2_scaled, y2_scaled)], outline=color, width=thickness)
    
    return np.array(pil_image)


def check_qa_pairs(info_file: str, view_index: int):
    """Check QA pairs for a specific info file and view index."""
    import matplotlib.pyplot as plt
    
    # Find corresponding image file
    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]
    
    # Visualize detections
    annotated_image = draw_detections(str(image_file), info_file)
    
    # Display the image
    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"View {view_index} - Red=Ego Car, Green=Other Karts")
    plt.show()
    
    # Generate QA pairs
    qa_pairs = generate_qa_pairs(info_file, view_index)
    
    # Print QA pairs
    print("\nQuestion-Answer Pairs:")
    print("-" * 50)
    for qa in qa_pairs:
        print(f"Q: {qa['question']}")
        print(f"A: {qa['answer']}")
        print("-" * 50)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'check':
        import fire
        fire.Fire({'check': check_qa_pairs})
    else:
        main()
