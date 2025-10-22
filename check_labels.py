"""Check what class IDs are in the dataset."""
from pathlib import Path

for split in ['train', 'valid', 'test']:
    label_dir = Path(f'data/DamagedRoadSigns/DamagedRoadSigns/{split}/labels')
    class_ids = set()
    
    for label_file in label_dir.glob('*.txt'):
        with open(label_file) as f:
            for line in f:
                if line.strip():
                    class_id = int(line.split()[0])
                    class_ids.add(class_id)
    
    print(f'{split:6s}: class IDs = {sorted(class_ids)}')

