import json
import os
import glob
from typing import List, Dict, Tuple
from tqdm import tqdm

TEXT_DIR = './dataset/HumanML3D/texts'        
INPUT_DIR  = './m2t_results'  
PATTERN    = 'm2t_results_*.json'                     

def get_txt_name(key: str) -> str:
    if '_' in key:
        return key.split('_', 1)[1]
    return key


def parse_txt(path: str) -> List[Dict]:
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('#')
            if len(parts) < 4:
                continue
            caption = parts[0].strip()
            tokens = parts[1].strip()
            try:
                f_tag = float(parts[2]) if parts[2] != 'nan' else 0.0
                to_tag = float(parts[3]) if parts[3] != 'nan' else 0.0
            except ValueError:
                f_tag = to_tag = 0.0
            lines.append({
                'caption': caption,
                'tokens': tokens,
                'f_tag': f_tag,
                'to_tag': to_tag,
            })
    return lines


def build_caption_list(records: List[Dict], target_f: float, target_t: float) -> List[str]:
    filtered = [r['caption'] for r in records
                if abs(r['f_tag'] - target_f) < 1e-6 and abs(r['to_tag'] - target_t) < 1e-6]

    if len(filtered) >= 3:
        return filtered[:3]
    elif len(filtered) == 2:
        return [filtered[0], filtered[1], filtered[0]]
    elif len(filtered) == 1:
        return [filtered[0]] * 3
    else:
        return [''] * 3  


def process_one_file(input_json: str):
    output_json = input_json.replace('.json', '_match.json')

    with open(input_json, 'r', encoding='utf-8') as f:
        data: Dict[str, Dict] = json.load(f)

    for key, val in tqdm(data.items(), desc=os.path.basename(input_json)):
        caption_to_match = val.get('caption', '').strip()
        if not caption_to_match:
            continue

        txt_name = get_txt_name(key)
        txt_path = os.path.join(TEXT_DIR, txt_name + '.txt')
        if not os.path.isfile(txt_path):
            print(f'Warning: {txt_path} not found, skip.')
            continue

        records = parse_txt(txt_path)

        target_ft: Tuple[float, float] = None
        for r in records:
            if r['caption'] == caption_to_match:
                target_ft = (r['f_tag'], r['to_tag'])
                break

        if target_ft is None:
            print(f'Warning: caption not found in {txt_path}, skip key {key}')
            continue

        caption_list = build_caption_list(records, *target_ft)
        val['caption_list'] = caption_list

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print('Done. Output saved to', output_json)


def main():
    input_jsons = sorted(glob.glob(os.path.join(INPUT_DIR, PATTERN)))
    if not input_jsons:
        print('No files matched pattern:', os.path.join(INPUT_DIR, PATTERN))
        return

    for json_file in input_jsons:
        print('Processing:', json_file)
        process_one_file(json_file)

    print('All files processed.')


if __name__ == '__main__':
    main()