## Insightface_arcface_pytorch

This repo just recorded the project I have made last few months. The project uses  ArcFace, and based on Arcface, I provide `dataset_prepare.py` for own dataset prepareing, and also provide `create_pairs.py` to create `pairs.txt` for testing.

### Requirements

```
numpy
torch
PIL
torchvision
tqdm
splitfolders
````

### Dataset prepare

1. First put your photos in folders respectively. The structure just like the following

```
A(folder)
    12443.jpg
    423435.jpg
    ...
B (folder)
    64344.jpg
    34353.jpg
    ...
```
2. Run `python dataset.prepare.py`


3. After running, the structure may become 
the following
```
照片資料夾-
    資料夾A
            - A_00.jpg
            - A_01.jpg
    資料夾B
            - B_00.jpg
            - B_01.jpg
            ...
            
    train(folder，每個folder中至少放一張沒戴口照照片)
            -資料夾A
            - A_00.jpg
            - A_09.jpg(照片數量是原本的7成)
            ...
    val(folder)
            -資料夾A
            - A_02.jpg
            - A_04.jpg(照片數量是原本的3成)
            ...
# If you want model could recognize pepole with mask, you have to put at least one no masked photo in each folder in train folder manually
```
As seen, train folder(70% photos of original photos) and val folder(30% photos of original photos) are created, and there are photos in each folder.

### Train

1. Run `python train.py`
2. After training, there may be a `23.pth `file in checkpoints folder

### Test

1. First create pairs.txt for testing. Run `python create_pairs.py`
2. Make sure pairs.txt exists and is normal like this(1 stands for the same person, 0 stands for other person)

```
04\04_5.jpg 04\04_1.jpg 1
40\40_3.jpg 40\40_7.jpg 1
05\05_1.jpg 05\05_0.jpg 1
27\27_2.jpg 27\27_3.jpg 1
34\34_7.jpg 30\30_3.jpg 0
49\49_2.jpg 13\13_8.jpg 0
36\36_4.jpg 60\60_0.jpg 0
...
```
3. Run `test_model.py` 

4. Finally, you will get `accuracy`,`threshold`

### Contribution and Restriction

1.[arcface-pytorch](https://github.com/ronghuaiyang/arcface-pytorch)
2.[Build-Your-Own-Face-Model](https://github.com/ColinFred/Build-Your-Own-Face-Model): there is a mistake on comment( actually is not `distance` ,but `similarity`)
* for more , please check
 https://blog.csdn.net/weixin_43977640/article/details/115579153
https://www.cnblogs.com/suanec/p/9121092.html

