import cv2
import os
import random
from tqdm import tqdm

cats = [os.path.join("./temp_data/cat", l) for l in os.listdir("./temp_data/cat")]
notcats = [os.path.join("./temp_data/notcat_sample", l) for l in os.listdir("./temp_data/notcat_sample")]

selected_notcats = [notcats[random.randint(0, len(notcats) - 1)] for _ in range(len(cats))]

for i, notcat in tqdm(enumerate(selected_notcats)):
    img = cv2.imread(notcat)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    x, y = (random.randint(0, img.shape[i] - 2001) for i in range(2))

    img_part = img[x:(x+2000), y:(y+2000), :]

    cv2.imwrite("temp_data/notcat/" + str(i) + ".jpg", img_part)

print(notcats)
