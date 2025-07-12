import sys
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
import pickle
from tqdm import tqdm

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))

def write_masks_to_png(masks, image, path: str) -> None:
    plt.figure(figsize=(20,20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    #plt.show()
    plt.savefig(path)
    return

    

sam_checkpoint = "/home/phucnda/applied_cv/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
mask_generator = SamAutomaticMaskGenerator(sam)

# with open('/root/3dllm/data/scannetpp/metadata/instance_classes.txt', 'r') as file:
#     instance_classes = file.readlines()
#     instance_classes = [instance.strip() for instance in instance_classes]

# with open('/root/3dllm/data/scannetpp/metadata/semantic_classes.txt', 'r') as file:
#     semantic_classes = file.readlines()
#     semantic_classes = [semantic.strip() for semantic in semantic_classes]



with open('/home/phucnda/applied_cv/OmniScient-Model/dataset3d/scannet200_val.txt', 'r') as file:
    scenes = file.readlines()
    scenes = [scene.strip() for scene in scenes]


path2dmasks = '/home/phucnda/applied_cv/Dataset/sam2d_scannet200'

# phucnda1
# 0 : 0:30 , 30:60
# 1 : 60:90, 90:120

# phucnda
# 0 : 120:150 , 150:180
# 1 : treo
for scene_id in tqdm(scenes[150:180]):
    # extract image from video
    output_folder_image = os.path.join("/home/phucnda/applied_cv/Dataset/ScannetV2/ScannetV2_2D_5interval/trainval", scene_id, "color")

    save2d_mask_folder = os.path.join(path2dmasks,scene_id)
    
    image_ids = sorted(os.listdir(output_folder_image))
    for image_id in tqdm(image_ids):
        image_path = os.path.join(output_folder_image, image_id)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(image)
        

        os.makedirs(os.path.join(save2d_mask_folder, "weights"), exist_ok=True)
        os.makedirs(os.path.join(save2d_mask_folder, "images"), exist_ok=True)

        save2d_mask_path = os.path.join(save2d_mask_folder, "weights", image_id.split(".")[0] + ".pkl")
        saveimg_mask_path = os.path.join(save2d_mask_folder, "images", image_id.split(".")[0] + ".png")
        
        results = {}
        for key in masks[0].keys():
            if key == "segmentation":
                results["pred_masks"] = np.stack([mask[key] for mask in masks])
            elif key == "stability_score":
                results["scores"] = np.stack([mask[key] for mask in masks])
            else:
                results[key] = np.stack([mask[key] for mask in masks])
        
        # Writing to the file using pickle.dump()
        with open(save2d_mask_path, 'wb') as file:
            pickle.dump(results, file)

        # write_masks_to_png(masks=masks, image=image, path=saveimg_mask_path)
        # breakpoint()
    print("Completed scene ", scene_id)