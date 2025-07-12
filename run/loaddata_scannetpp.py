import torch
import numpy as np
import cv2
from tqdm import tqdm, trange
from PIL import Image
import sys; sys.path.append('/root/OmniScient-Model')
import os
import matplotlib.pyplot as plt
import torchvision.transforms as T
# Dataloader
from loader3d.loader import ScannetReader, align_pixel
from loader3d.loaderpp import ScanNetPPReader
from loader3d.loader_util import PointCloudToImageMapper, resolve_overlapping_masks, show_mask
# Detic
import pickle
from detectron2.structures import Instances
import pycocotools.mask
from typing import Union, Dict
# OSM LLM
import torchvision.transforms as T
from utils import prepare_sam, prepare_osm, prepare_instruction, prepare_image_from_image, prepare_image, get_masks, get_context_mask, get_classes, get_classes_batch, get_classes_multibatch

def read_detectron_instances(filepath: Union[str, os.PathLike], rle_to_mask=True) -> Instances:
    with open(filepath, 'rb') as fp:
        instances = pickle.load(fp)
        if rle_to_mask:
            if instances.pred_masks_rle:
                pred_masks = np.stack([pycocotools.mask.decode(rle) for rle in instances.pred_masks_rle])
                instances.pred_masks = torch.from_numpy(pred_masks).to(torch.bool)  # (M, H, W)
            else:
                instances.pred_masks = torch.empty((0, 0, 0), dtype=torch.bool)
    return instances

def calculate_iou(a, b):
    intersection = torch.logical_and(a, b)
    union = torch.logical_or(a, b)
    iou = torch.sum(intersection) / torch.sum(union)
    return iou

@torch.no_grad()
def post_processing_to_panoptic_mask(masks, classes, class_probs, class_threshold=0.2,
                    overlapping_ratio=0.8, mask_min_size=100):
    assert len(masks) == len(classes) and len(classes) == len(class_probs)

    # post-processing as in kMaX-DeepLab style, to obtain non-overlapping masks (i.e., panoptic masks)
    class_probs = torch.tensor(class_probs)
    reorder_indices = torch.argsort(class_probs, dim=-1, descending=True)
    pan_mask = np.zeros_like(masks[0]).astype(np.uint8)
    final_classes = []
    new_idx = 1

    for i in range(len(masks)):
        cur_idx = reorder_indices[i].item() # 1
        cur_mask = masks[cur_idx]
        cur_class = classes[cur_idx]
        cur_prob = class_probs[cur_idx].item()
        if cur_prob < class_threshold:
            continue
        assert cur_class
        original_pixel_num = cur_mask.sum()
        new_binary_mask = np.logical_and(cur_mask, pan_mask==0)
        new_pixel_number = new_binary_mask.sum()
        if original_pixel_num * overlapping_ratio > new_pixel_number or new_pixel_number < mask_min_size:
            continue
        pan_mask[new_binary_mask] = new_idx
        final_classes.append(cur_class)
        new_idx += 1

    return pan_mask, final_classes, new_idx

def rle_decode(rle):
    """Decode rle to get binary mask.

    Args:
        rle (dict): rle of encoded mask
    Returns:
        mask (np.ndarray): decoded mask
    """
    length = rle["length"]
    s = rle["counts"]

    starts, nums = [np.asarray(x, dtype=np.int32) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + nums
    mask = np.zeros(length, dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask

query_model_osm = None

class freevocab():
    def __init__(self, mask3d, mask2d, pcl, data2d, gtdata, pathdata2d, interval, osm_checkpoint, scene_id, pointdata):
        
        self.frames = [] ### Series of frames looping by interval step
        self.proposal3d = None ### np array of 3D class-agnostic proposals
        self.conf3d = None ### np array of 3D class-agnostic proposals confidence score
        self.model = query_model_osm
        self.scene_id = scene_id
        #### Set up dataloader ####
        try:
            loader = ScanNetPPReader(root_path = data2d + '/iphone')
        except FileNotFoundError:
            print('>>> Error: Data not found. Did you download it?')
            exit()

        print("Number of frames to load: ", len(loader))
        
        #### Point2Image mapper ####
        img_dim = (1920,  1440)
        cut_num_pixel_boundary = 10 # do not use the features on the image boundary
        self.point2img_mapper =  PointCloudToImageMapper(
            image_dim=img_dim, intrinsics=None, cut_bound=10
        )
        self.point, self.color, self.sem_label, self.ins_label = torch.load(gtdata) # compressed point
        self.point, self.color, _ , _ = torch.load(pointdata) # raw point
        mask3d = torch.load(mask3d)

        self.mask2d = mask2d
        mask_data_dict_goc = torch.load(self.mask2d)

        for i in trange(0, len(loader),interval * 5):
            frame = loader[i]
            frame_id = frame['frame_id']  # str
            if frame_id not in mask_data_dict_goc.keys():
                continue
            mask_data_dict = mask_data_dict_goc[frame_id]
            pose, instrinsic = loader.read_pose(frame["pose_path"], frame["raw_frame_id"])
            depth = loader.read_depth(frame["depth_path"])
            depth = cv2.resize(depth, (img_dim[0], img_dim[1]))

            image_sam = loader.read_image(frame["image_path"])
            image_path = frame['image_path']  # str

            #### Point mapping ####
            n_points = self.point.shape[0]
            mapping = np.ones([n_points, 4], dtype=int)
            mapping[:, 1:4] = self.point2img_mapper.compute_mapping(pose, self.point, depth, intrinsic=instrinsic)
            mapping = torch.tensor(mapping)
            
            idx = torch.where(mapping[:,3] == 1)[0]
            # no points corresponds to this image, visible points on 2D image
            if mapping[:, 3].sum().item() == 0 or idx.shape[0]<100: 
                continue

            encoded_masks = mask_data_dict["masks"]
            masks = None
            if encoded_masks is not None:
                masks = []
                for mask in encoded_masks:
                    try:
                        tmp = torch.tensor(pycocotools.mask.decode(mask))
                    except:
                        continue
                    masks.append(tmp)
                masks = torch.stack(masks, dim=0).cpu()  # cuda fast but OOM

            #### Record Information ####
            dic = {}
            dic['image'] = image_sam
            dic['mapping'] = mapping
            dic['masks'] = masks
            self.frames.append(dic)
            #### Visualization of an image ####
            if False:
                # draw output image
                plt.figure(figsize=(10, 10))
                plt.imshow(image_sam)
                for mask in masks[:10]:
                    show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
                plt.axis('off')
                plt.savefig(os.path.join("devfig/detic"+str(i)+'.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)

        self.proposal3d = mask3d['ins']
        self.proposal3d = torch.tensor([rle_decode(ins) for ins in self.proposal3d])


if __name__=='__main__':
    
    # Load queryable model
    osm_checkpoint = '../weights/osm_final.pt'
    # query_model_osm = OSM(osm_checkpoint)
    
    ######################### PerScene #########################
    per_scene = True
    if per_scene == True:
        #### datapath
        scene_id = '0d2ee665be'
        path3dmask = '../Dataset/hier_agglo_scannetpp/hier_agglo' # change the path to open3dis 3D mask (open3dis_sam_val_200)
        path2dmask = '../Dataset/scanpp/scannetpp/masksam_val'

        pathpcl = '../Dataset/scanpp/scannetpp/processed_pcl_scannet200like' # Take XYZ, RGB (inst_no_stuff)
        pathdata2d = '../Dataset/scanpp/scannetpp/data' # scenes containing color/depth/pose
        pathgtdata = '../Dataset/scanpp/scannetpp/processed_pcl_scannet200like' # gt label of points following order compressed points
        pathpointdata = '../Dataset/scanpp/scannetpp/processed_pcl_scannet200like'
        interval = 2
        
        #### merging path
        mask3d = os.path.join(path3dmask, scene_id + '.pth')
        mask2d = os.path.join(path2dmask, scene_id +'.pth')
        pcl = os.path.join(pathpcl, scene_id + '.pth')
        data2d = os.path.join(pathdata2d, scene_id)
        gtdata = os.path.join(pathgtdata, scene_id + '.pth')
        pointdata = os.path.join(pathpointdata, scene_id + '.pth')
        #### Processing
        container = freevocab(mask3d, mask2d, pcl, data2d, gtdata, pathdata2d, interval, osm_checkpoint, scene_id, pointdata)
    ###########################################################################
