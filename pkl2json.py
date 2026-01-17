import pickle
import json

file_path = "./dataset/HumanML3D/tmp_think/data_dict_train.pkl"
output_json = "motion_train.json"  

with open(file_path, 'rb') as f:
    data = pickle.load(f)

result = []

for name, sample in data.items():
    tokens = sample['motion']
    tokens_str = ''.join([f'<Motion_{x}>' for x in tokens])
    motion_tokens_text = f'<Motion>{tokens_str}</Motion>'
    
    for text_item in sample['text']:
        result.append({
            "name": name.split('_')[-1] if '_' in name else name,
            "caption": text_item['caption'],
            "tokens": text_item['tokens'],
            "motion_think": text_item['motion_think'],
            "motion": motion_tokens_text,
            "f_tag": text_item['f_tag'],
            "to_tag": text_item['to_tag'],
        })

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(result[0])


import json

with open('./dataset/HumanML3D/train.txt', 'r') as f:
    train_names = {line.strip() for line in f.readlines()} 

with open('motion_train.json', 'r') as f:
    motion_data = json.load(f)

motion_names = {entry['name'] for entry in motion_data}

missing_names = motion_names - train_names  

if missing_names:
    print("missing_names:", missing_names)
else:
    print("All names exist")

