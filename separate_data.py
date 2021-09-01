import os
import random
from tqdm import tqdm
from shutil import copyfile

cats = [os.path.join("./temp_data/cat", l) for l in os.listdir("./temp_data/cat")]
notcats = [os.path.join("./temp_data/notcat", l) for l in os.listdir("./temp_data/notcat")]

train_cats = [cats[i] for i in random.sample(range(len(cats)), int(float(len(cats) * 0.9)))]
train_notcats = [notcats[i] for i in random.sample(range(len(notcats)), int(float(len(notcats) * 0.9)))]

test_cats = [c for c in cats if c not in train_cats]
test_notcats = [n for n in notcats if n not in train_notcats]

for i, cat in tqdm(enumerate(train_cats)):
    copyfile(cat, "data/train/cat/" + str(i) + ".jpg")

for i, cat in tqdm(enumerate(test_cats)):
    copyfile(cat, "data/test/cat/" + str(i) + ".jpg")

for i, notcat in tqdm(enumerate(train_notcats)):
    copyfile(notcat, "data/train/notcat/" + str(i) + ".jpg")

for i, notcat in tqdm(enumerate(test_notcats)):
    copyfile(notcat, "data/test/notcat/" + str(i) + ".jpg")


print("here")