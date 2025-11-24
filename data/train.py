import json
from pathlib import Path
from homework.generate_qa import generate_qa_pairs

def generate_all_qa_pairs(split='train'):
    """Generate QA pairs for all frames in the dataset."""
    data_dir = Path(f'/content/adl_h4/data/{split}')
    
    # Find all info.json files
    info_files = sorted(data_dir.glob('*_info.json'))
    
    print(f"Found {len(info_files)} info files in {split}")
    
    all_qa_pairs = []
    
    for info_file in info_files:
        frame_id = info_file.stem.replace('_info', '')
        
        # Generate QA pairs for all 10 views (0-9)
        for view_index in range(10):
            try:
                qa_pairs = generate_qa_pairs(str(info_file), view_index)
                
                # Add image file path to each QA pair
                for qa in qa_pairs:
                    qa['image_file'] = f"{split}/{frame_id}_{view_index:02d}_im.jpg"
                
                all_qa_pairs.extend(qa_pairs)
                
            except Exception as e:
                print(f"Error processing {frame_id} view {view_index}: {e}")
                continue
        
        if len(info_files) > 100 and info_file == info_files[0]:
            print(f"Processed {frame_id}, continuing...")
    
    print(f"Generated {len(all_qa_pairs)} total QA pairs")
    
    # Save to JSON
    output_file = data_dir / 'balanced_qa_pairs.json'
    with open(output_file, 'w') as f:
        json.dump(all_qa_pairs, f, indent=2)
    
    print(f"Saved to {output_file}")
    return all_qa_pairs

# Generate for training data
qa_pairs = generate_all_qa_pairs('train')