import os
import glob


def main():
    for i in range(2, 50):
        if os.path.exists(f'data/experimentdata{i}'):
            generate('data/experimentdata1/', f'data/experimentdata{i}')
            print(f'processed data/experimentdata{i}')
    
def generate(los_dir: str, mpath_dir: str):
    target_dir = mpath_dir.replace('experimentdata', 'p')
    os.makedirs(target_dir, exist_ok=True)
    for i in range(1, 33):
        los_path = os.path.join(los_dir, f'{i}.txt')
        mpath_path = os.path.join(mpath_dir, f'{i}.txt')
        target_path = os.path.join(target_dir, f'{i}.txt')
        los_data = parse_file(los_path)
        mpath_data = parse_file(mpath_path)
        with open(target_path, 'w') as f:
            for los_line, mpath_line in zip(los_data, mpath_data):
                los_freq, los_amplitude, los_phase = los_line
                mpath_freq, mpath_amplitude, mpath_phase = mpath_line
                f.write(f'{los_freq},{float(mpath_amplitude)/float(los_amplitude)},{float(mpath_phase)/float(los_phase)}\n')

def parse_file(file_path: str):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        data.append(line.strip().split(','))
    return data

if __name__ == '__main__':
    main()