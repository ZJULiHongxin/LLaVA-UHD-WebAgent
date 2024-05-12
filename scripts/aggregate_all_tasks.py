import os, json, glob
import random
random.seed(666)
from tqdm import tqdm
from pprint import pprint
import cv2
task_dir = "/data0/jingran/workspace/UI_training_data/Ours-Pretrain"

EXAMPLE = False
USE_TAG = True

# ER代表element refer，AG代表action grounding，mc代表multiple choice，box代表对应的图片里圈了红色的bbox，som代表多个带颜色的bbox，tag代表bbox文本前后加了<>

# 仅包含grounding任务，
SELECTED_TASK_TYPES = ["AG_point"]
EXCLUDED_TASK_TYPES = [""]

aggr_name = "mixed_AG_point_tag.json"

skipped = ["seeclick_web.json", "NEG"]
only = [] #["omniact"]
aggr = []

task_stats = {}

all_tasks = glob.glob(os.path.join(task_dir, "*.json"))
for task in tqdm(all_tasks, total=len(all_tasks)):
    if aggr_name in task: continue
    
    if (len(only) > 0 and not any(k in task for k in only)) or (len(SELECTED_TASK_TYPES) and not any(k in task for k in SELECTED_TASK_TYPES)) or any(k in task for k in skipped) or (USE_TAG and '_tag' not in task):
        print("Skip", task)
        continue

    task_name = os.path.basename(task)[:-5]
    
    print("Processing", task_name)
    with open(task, "r") as f:
        samples = json.load(f)
    
    
    if 'refexp_' in task_name:
        samples = random.sample(samples, len(samples) // 3)

    if EXAMPLE:
        samples = samples[0:1]

    task_stats[task_name] = len(samples)

    for sample_idx, sample in enumerate(samples):
        sample["id"] = f"{task_name}#{sample['id']}"
        
        for message_id in range(1, len(sample["conversations"]), 2):
            answer = sample["conversations"][message_id]["value"]
            bbox_start, bbox_end = answer.find("<bbox>["), answer.find("]<")
            
            try:
                if bbox_start == -1:
                    point_start, point_end = answer.find("<point>("), answer.find(")<")
                    centerX, centerY = list(map(float, answer[point_start + 8: point_end].split(',')))
                    
                    assert point_end != -1, f"invalid: {answer}"
                    
                    if centerX > 1000 or centerY > 1000:
                        img = cv2.imread(os.path.join("/data0/jingran/workspace/UI_training_data/Ours-Pretrain/images", sample["image"]), cv2.IMREAD_COLOR)
                    
                        centerX, centerY = centerX / img.shape[1] * 1000, centerY / img.shape[0] * 1000
                                    
                    new_answer = answer[:point_start].replace("within the", "at the point").replace('"bbox":', '"location":') + f"<point>({int(centerX):03d}, {int(centerY):03d})</point>" + answer[point_end+9:]
                else:
                    assert bbox_end != -1, f"invalid: {answer}"
                    bbox_coords = list(map(int, answer[bbox_start + 7: bbox_end].split(',')))

                    centerX, centerY = (bbox_coords[0]+bbox_coords[2]) / 2, (bbox_coords[2]+bbox_coords[3]) / 2
                    
                    new_answer = answer[:bbox_start].replace("within the bounding box", "at the point") + f"<point>({int(centerX):03d}, {int(centerY):03d})</point>" + answer[bbox_end+7:]
            except Exception as e:
                print("invalid:", answer)

            if sample_idx == 1:
                img = cv2.imread(os.path.join("/data0/jingran/workspace/UI_training_data/Ours-Pretrain/images", sample["image"]), cv2.IMREAD_COLOR)
                img = cv2.circle(img, (int(img.shape[1] * centerX / 1000), int(img.shape[0] * centerY / 1000)), radius=5,color=(0,255,0), thickness=2)
                
                img = cv2.putText(img, text=sample["conversations"][message_id-1]["value"], org=(20,20), fontScale=1, color=(0,255,0), fontFace=cv2.FONT_HERSHEY_SIMPLEX)
                
                input((int(img.shape[1] * centerX / 1000), int(img.shape[0] * centerY / 1000)))
                cv2.imwrite(f"{task_name}_{sample_idx}.png", img)

            sample["conversations"][message_id]["value"] = new_answer

        aggr.append(sample)

pprint(task_stats)
print(f"All: {len(aggr)}")
with open(os.path.join(task_dir, aggr_name), "w") as f:
    json.dump(aggr, f, indent=2)
