import json, os, re
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
data_file = os.path.join(os.path.dirname(__file__), "omniact_AG_point_tag.json")
'mixed_AG_point_tag.json'
with open(data_file, 'r') as f:
    data = json.load(f)

NUM_BINS = 50
coord_range = np.linspace(0, 1000, NUM_BINS+1)
XY_hist = np.ones((NUM_BINS, NUM_BINS))

XY_collections = [[[] for _ in range(NUM_BINS)] for _ in range(NUM_BINS)]
XY_counts = [[0 for _ in range(NUM_BINS)] for _ in range(NUM_BINS)]

def extract_points(text):
    x, y = list(map(int, text[text.find("<point>") + 8: text.find(")</point>")].split(',')))
    
    return x, y

for sample in tqdm(data):
    x, y = extract_points(sample['conversations'][-1]["value"])
    x, t = int(x), int(y)

    x_idx = np.digitize(x, coord_range) - 1
    y_idx = np.digitize(y, coord_range) - 1
    XY_collections[x_idx][y_idx].append(sample)
    XY_counts[x_idx][y_idx] += 1

max_count, max_X, max_Y = 0, 0, 0
for i in range(NUM_BINS):
    for j in range(NUM_BINS):
        if XY_counts[i][j] > max_count:
            max_count = XY_counts[i][j]
            max_X = i
            max_Y = j

print(f"Max count: {max_count}, at ({max_X * 1000 // NUM_BINS}-{(max_X+1) * 1000 // NUM_BINS}, {max_Y * 1000 // NUM_BINS}-{(max_Y+1) * 1000 // NUM_BINS})")
# Draw the 2D distribution of X, Y coords
f = plt.figure()
plt.imshow(XY_counts, cmap='hot', interpolation='nearest', extent=[0, 1000, 0, 1000], origin='lower')
plt.colorbar(label='Count')  # Add a color bar to show count scale
plt.title('2D Distribution of X, Y Coordinates')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(False)  # Turn off the grid if not desired
plt.savefig("sample_dist.png")