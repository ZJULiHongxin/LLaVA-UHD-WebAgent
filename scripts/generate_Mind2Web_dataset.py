import os, json, tqdm, random, cv2
import numpy as np
random.seed(1234)
from bs4 import BeautifulSoup
from datasets import load_dataset

our_data_dir = "/data0/jingran/workspace/UI_training_data/Ours-Pretrain"
image_dir = os.path.join(our_data_dir, "images/mind2web")

os.makedirs(image_dir, exist_ok=True)

dataset = load_dataset("osunlp/Multimodal-Mind2Web")

MAX_W, MAX_H = 1280, 720
NUM_NEAREST = 6

NONE_STR = "A. None of the following"

SYS_PROMPT = """You are a helpful assistant that is great at website design, navigation, and executing tasks for the user."""

PROMPT = """Please choose the next move according to the UI screenshot, task instruction and previous actions.

Task Instruction: {task}. 

Previous actions: 
{prev_actions}

***IMPORTANT NOTES**
1. Firstly, Select the proper element from the following (If the correct element is not in the page above, please select {none_str}):
{choices}

2. Then select only one proper action from [CLICK, TYPE, SELECT] to interact with the selected element. Here is the action usage:
- Action "CLICK" should be formulated as json schema: '{{"element": str, "action_type": "CLICK"}}'. "element" (A,B,C,D,E,F) denotes the index of the selected element.
- Action "TYPE" should be followed with "value" (the typed texts), in json scehma: '{{"element": str, "action_type": "TYPE", "value": "..." }}'
- Action "SELECT" should be followed with "value", in json schema: '{{"element": str, "action_type": "SELECT", "value": "..." }}'. "value" denotes the option selected from the candidates shown in the screenshot.
3. The answer should be concise and formulated following above json schema.
***IMPORTANT NOTES End**

Your answer:"""

def get_bbox(text):
    x1, y1, W, H = list(map(lambda x: int(float(x)), text.split(',')))
    return [x1,y1,x1+W,y1+H]

def is_overlap(rect1, rect2, th=0.2):
    # Unpack the coordinates
    x1_A, y1_A, x2_A, y2_A = rect1
    x1_B, y1_B, x2_B, y2_B = rect2
    
    # Calculate the coordinates of the intersection rectangle
    x1_inter = max(x1_A, x1_B)
    y1_inter = max(y1_A, y1_B)
    x2_inter = min(x2_A, x2_B)
    y2_inter = min(y2_A, y2_B)
    
    # Calculate the area of intersection
    if x1_inter < x2_inter and y1_inter < y2_inter:
        intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    else:
        intersection_area = 0
    
    # Calculate the area of both rectangles
    area_A = (x2_A - x1_A) * (y2_A - y1_A)
    area_B = (x2_B - x1_B) * (y2_B - y1_B)
    
    if area_A < 1: return True
    else: return intersection_area / area_B > th or intersection_area / area_A > th

sample_collections = []

for i, sample in tqdm.tqdm(enumerate(dataset["train"])):
    if i > 10: break

    # Load the HTML code
    soup = BeautifulSoup(sample["raw_html"], "html.parser")
    
    # Collect the bboxes of every node
    elem_texts, bboxes, node_ids = [], [], []
    
    pos_cand = sample["pos_candidates"][0]
    pos_cand_attrs = json.loads(json.loads(pos_cand)["attributes"])
    pos_cand_node_id, pos_cand_box = pos_cand_attrs["backend_node_id"], get_bbox(pos_cand_attrs['bounding_box_rect']) # x1, y1, x2, y2

    def inspect_leaf_nodes(element):
        # Recursive function to find and inspect leaf nodes
        if element.attrs.get("backend_node_id", "") == pos_cand_node_id:
            elem_texts.append(str(element).split('\n')[0])
            node_ids.append(element.attrs["backend_node_id"])
            bboxes.append(get_bbox(element.attrs.get("bounding_box_rect", None)))

        if element.find_all(recursive=False):
            # If the element has children, recurse into them
            for child in element.find_all(recursive=False):
                inspect_leaf_nodes(child)
        elif hasattr(element, 'attrs'):# No children, this is a leaf node
            box = element.attrs.get("bounding_box_rect", None)

            if box is not None:
                elem_texts.append(str(element))
                node_ids.append(element.attrs["backend_node_id"])
                bboxes.append(get_bbox(box))
    
    
    # Start from the root element
    inspect_leaf_nodes(soup)
    
    # save the image surrounding the pos candidate
    if len(sample["pos_candidates"]) == 0: continue


    if pos_cand_node_id not in node_ids:
        continue
        
    upper = max(0, pos_cand_box[1] - MAX_H)
    lower = upper + MAX_H
    
    # crop the image
    cropped_image = sample["screenshot"].crop([0, upper, MAX_W, lower])
    
    img_path = os.path.join(image_dir, f"{sample['annotation_id']}_{sample['action_uid']}.png")
    cropped_image.save(img_path)
    
    # Adjust the boxes of pos/neg candidates
    pos_cand_box = [pos_cand_box[0], max(0, pos_cand_box[1] - upper), pos_cand_box[2], min(MAX_H, pos_cand_box[3] - upper)]

    # Retain elements inside the view
    neg_cands_ids = [json.loads(json.loads(neg)["attributes"])['backend_node_id'] for neg in sample["neg_candidates"]]

    bboxes_inview, elemtext_inview, node_ids_inview, neg_cand_ids_inview, neg_cand_elemtext_inview, neg_cand_boxes_inview = [], [], [], [], [], []
    pos_cand_elemtext = None

    for j in range(len(node_ids)):
        box = bboxes[j]
        box_h = box[3] - box[1]
        if box[1] < upper and (upper - box[3]) <= 0.6 * box_h or box[1] >= lower and (box[3] - lower) <= 0.6 * box_h or box[1] >= lower or box[3] < upper: continue

        adjusted_box = [box[0], max(0, box[1] - upper), box[2], min(MAX_H, box[3] - upper)]
        bboxes_inview.append(adjusted_box)
        elemtext_inview.append(elem_texts[j])
        node_ids_inview.append(node_ids[j])
        
        if node_ids[j] == pos_cand_node_id:
            pos_cand_elemtext = elem_texts[j]

        if node_ids[j] in neg_cands_ids:
            neg_cand_boxes_inview.append(adjusted_box)
            neg_cand_elemtext_inview.append(elem_texts[j])
            neg_cand_ids_inview.append(node_ids[j])

    if len(neg_cand_ids_inview) == 0: continue

    # The answer is A. None at p=0.5
    none_answer = random.random() < 0.5
    
    # Select negative candidates
    selected_neg_idxs = random.sample(range(len(neg_cand_ids_inview)), k=min(len(neg_cand_ids_inview), 5 if none_answer else 4))
    selected_neg_cand_boxes = [neg_cand_boxes_inview[j] for j in selected_neg_idxs]
    selected_neg_cand_elemtext = [neg_cand_elemtext_inview[j] for j in selected_neg_idxs]
    
    # Find the nearest 6 elements to the pos/neg candidates
    all_boxes_inview = np.array(bboxes_inview)
    cands = np.array([pos_cand_box] + selected_neg_cand_boxes)
    
    all_box_centers = np.concatenate([all_boxes_inview[:, [0,2]].mean(axis=1, keepdims=True), all_boxes_inview[:, [1,3]].mean(axis=1, keepdims=True)], axis=1) # M x 2
    
    cands_centers = np.concatenate([cands[:, [0, 2]].mean(axis=1, keepdims=True), cands[:, [1, 3]].mean(axis=1, keepdims=True)], axis=1) # N x 2
    
    dists = np.linalg.norm(np.expand_dims(all_box_centers, 1) - np.expand_dims(cands_centers, 0), axis=2) # N x M
    
    # Find the indices the of the nearest 6 neighboring elements
    nearest_indices = np.argsort(dists, axis=0)[:, :6] # N x 6
    nearest_elemtexts, nearest_boxes, nearest_idxs = [[] for _ in range(len(cands))], [[] for _ in range(len(cands))], [[] for _ in range(len(cands))]
    
    for cand_id, cand_box in enumerate(cands):
        pointer = 0
        while pointer < nearest_indices.shape[0] and len(nearest_boxes[cand_id]) < NUM_NEAREST:
            # Select a random index
            this_idx = nearest_indices[pointer, cand_id]
            box = all_boxes_inview[this_idx].tolist()
            
            if not is_overlap(box, cand_box):
                nearest_idxs[cand_id].append(this_idx)
                nearest_boxes[cand_id].append(box)
                nearest_elemtexts[cand_id].append(elemtext_inview[this_idx])
            pointer += 1
    

    # Vis
    if False:
        screenshot = np.asarray(cropped_image).copy()
        # Draw the pos cand and its nearest 6 neighbors using their boxes
        for cand_id, cand_box in enumerate(cands):
            for box_id, nearest_box in enumerate(nearest_boxes[cand_id]):
                screenshot = cv2.rectangle(screenshot, (int(nearest_box[0]), int(nearest_box[1])), (int(nearest_box[2]), int(nearest_box[3])), (0, 0, 255), 2)
                screenshot = cv2.putText(screenshot, str(box_id), (int(nearest_box[0]), int(nearest_box[1])), color=(0, 0, 255), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2)
            screenshot = cv2.rectangle(screenshot, (int(cand_box[0]), int(cand_box[1])), (int(cand_box[2]), int(cand_box[3])), (0, 255, 0) if cand_id == 0 else (0, 128, 0), 2)
            
            screenshot = cv2.putText(screenshot, f"#{str(cand_id)}", (int(cand_box[0]), int(cand_box[1])), color=(128, 0, 0), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2)
        cv2.imwrite(f"{sample['annotation_id']}_{sample['action_uid']}_vis.png", screenshot)
    

    # # Make candidates
    cands_text = f"{NONE_STR}\n"
    
    if not none_answer:
        pos_rand_idx = random.randint(0, len(selected_neg_cand_elemtext))
        selected_neg_cand_elemtext.insert(pos_rand_idx, pos_cand_elemtext)
        pos_option_id = pos_rand_idx + 1
        
        sample_op_attrs = json.loads(sample["operation"])
        action, value = sample_op_attrs["op"], sample_op_attrs["value"]
        if action == "CLICK":
            action_str = ', "action_type": "CLICK"'
        elif action in ["SELECT", "TYPE"]:
            action_str = f', "action_type": "{action}", "value": "{value}"'

    else:
        pos_option_id = 0
        action_str = ""

    cands_text += "\n".join([f"{chr(j + 66)}. {neg_elemtext}" for j, neg_elemtext in enumerate(selected_neg_cand_elemtext)])
    
    if cands_text.count("None") > 1:
        print()
        continue

    target_action_index = int(sample["target_action_index"])
    if target_action_index == 0:
        prev_actions = 'None'
    else:
        prev_actions = '\n'.join(sample["action_reprs"][:target_action_index])

    conversations = [
        {
            "from": "system",
            "value": SYS_PROMPT
        },
        {
            "from": "user",
            "value": PROMPT.format(
                task=sample["confirmed_task"],
                none_str=NONE_STR,
                prev_actions=prev_actions,
                choices=cands_text)
        },
        {
            "from": "assistant",
            "value": f"""{{"element": "{chr(pos_option_id + 65)}"{action_str}}}"""
        }
    ]
    
    sample_collections.append(
        {
            "id": f"{sample['annotation_id']}_{sample['action_uid']}",
            "image": '/'.join(img_path.split('/')[-2:]),
            "conversation": conversations,
            "pos": {
                "option_id": pos_option_id,
                "box": pos_cand_box,
                "text": pos_cand_elemtext,
                "nearest_boxes": nearest_boxes[0],
                "nearest_texts": nearest_elemtexts[0],
            },
            "neg": [
                {
                    "box": neg_box,
                    "text": neg_elemtext,
                    "nearest_boxes": nearest_bs,
                    "nearest_texts": nearest_ts
                }
                for neg_box, neg_elemtext, nearest_bs, nearest_ts in zip(selected_neg_cand_boxes, selected_neg_cand_elemtext, nearest_boxes[1:], nearest_elemtexts[1:])
            ]
        }
    )
    
with open(os.path.join(our_data_dir, "mind2web.json"), "w") as f:
    print(f"Saving {len(sample_collections)} samples.")
    json.dump(sample_collections, f, indent=2)
