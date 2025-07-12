import torch
import numpy as np
import cv2
from tqdm import tqdm, trange
from PIL import Image
# import sys; sys.path.append('/root/OmniScient-Model')
import os
import matplotlib.pyplot as plt
import torchvision.transforms as T
# Dataloader
from loader3d.loader import ScannetReader, align_pixel
from loader3d.loader_util import PointCloudToImageMapper, resolve_overlapping_masks, show_mask
# Detic
import pickle
from detectron2.structures import Instances
import pycocotools.mask
from typing import Union, Dict
# OSM LLM
import torchvision.transforms as T
from utils import prepare_sam, prepare_osm, prepare_instruction, prepare_image_from_image, prepare_image, get_masks, get_context_mask, get_classes, get_classes_batch, get_classes_multibatch
from detectron2.structures import BitMasks
from torch.nn import functional as F
import pycocotools

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

class OSM():
    def __init__(self, osm_checkpoint):
        # Prepare OSM model and instructions
        self.class_generator, self.processor = prepare_osm(osm_checkpoint=osm_checkpoint)
        self.lang_x, self.qformer_lang_x = prepare_instruction(
            self.processor, "What is in the segmentation mask? Assistant:")
        self.input_size = self.processor.image_processor.size["height"]
        self.track = 0
    @torch.no_grad()
    def process_multiple_image(self, images, seg_masks, img_id, interval, imgs, highlight_masks):
        ## retrieve qformer embed ->  aggregate into pc feature

        image_for_osms = []
        for im in images:
            image = Image.fromarray(im).convert("RGB")
            if min(image.size) == max(image.size):
                image = T.functional.resize(image, size=self.input_size, interpolation=T.functional.InterpolationMode.BICUBIC)
            else: # in our case, always this case
                image = T.functional.resize(image, size=self.input_size - 1, max_size=self.input_size, interpolation=T.functional.InterpolationMode.BICUBIC)
        
            image_for_seg = np.array(image)
            # pad to input_size x input_size
            padded_image = np.zeros(shape=(self.input_size, self.input_size, 3), dtype=np.uint8)
            padded_image[:image_for_seg.shape[0], :image_for_seg.shape[1]] = image_for_seg
            image_for_osm = Image.fromarray(padded_image)  

            image_for_osms.append(image_for_osm)
        
        # batching the masks for not being OOM, we choose batch mask = 100
        segmentation_masks = []
        for batch_id in trange(0, len(seg_masks), 100):
            ### Generating class
            batch_end = min(batch_id + 100, len(seg_masks))
            segmentation = torch.nn.functional.interpolate(torch.stack(seg_masks[batch_id:batch_end]).to(torch.float16).unsqueeze(1).cuda(),size=(image.size[1], image.size[0]), mode='bicubic')
            segmentation = (segmentation > 0.5) # booling the mask
            segmentation = segmentation.squeeze(1).cpu() 
            segmentation_masks.append(segmentation)
            torch.cuda.empty_cache()

        # batch initialization
        images_batch = []
        segmentation_masks_batch = []
        context_mask_batch = []
        qformer_input_ids_batch = []
        qformer_attention_mask_batch = []
        input_ids_batch = []
        attention_mask_batch = []
        
        pred_class = []
        class_probs = []

        batch_size = 50
        number_imgs = len(image_for_osms)
        input_size = self.processor.image_processor.size["height"]
        # [N, 3, inpsize, inpsize] ~~
        torch.cuda.empty_cache()
          
        image = self.processor(images=image_for_osms, return_tensors="pt")["pixel_values"].view(number_imgs, 3, input_size, input_size)

        num_mask = 0
        for segs in tqdm(segmentation_masks):
            for seg in segs:
                binary_mask = seg
                padded_binary_mask = np.zeros(shape=(input_size, input_size), dtype=np.uint8)
                padded_binary_mask[:binary_mask.shape[0], :binary_mask.shape[1]] = binary_mask
                binary_mask = padded_binary_mask
                binary_mask = torch.from_numpy(np.ascontiguousarray(binary_mask.copy().reshape(1, input_size, input_size)))

                # if binary_mask.sum() < 100:
                #     pred_class.append("")
                #     class_probs.append(0)
                #     continue
                
                binary_mask = binary_mask.view(1, 1, input_size, input_size).float()
                
                images_batch.append(image[img_id[num_mask]])
                segmentation_masks_batch.append(binary_mask)
                context_mask_batch.append(get_context_mask(binary_mask, input_size, 0.5).view(
                    1, 1, input_size, input_size))
                qformer_input_ids_batch.append(self.qformer_lang_x["input_ids"])
                qformer_attention_mask_batch.append(self.qformer_lang_x["attention_mask"])
                input_ids_batch.append(self.lang_x["input_ids"])
                attention_mask_batch.append(self.lang_x["attention_mask"])
                num_mask += 1
        '''
        MULTI version: having batch size
        batch of image and mask  (1 image, n mask)
        duplicate into n(1 image, 1 mask) per forwarding

        img_id = [nMask], keeping track mask belong to which image
        '''
        qformer_embed = []
        print('Querying OSM...')
        for batch_id in trange(0, len(images_batch), batch_size):
            ### Generating class
            batch_end = min(batch_id + batch_size, len(images_batch))
            with torch.no_grad():
                generated_output_qformer = self.class_generator.generate_1n_avg_qformer(
                    img_idd = img_id[batch_id:batch_end],
                    pixel_values=torch.stack(images_batch[batch_id:batch_end]).cuda().to(torch.float16),
                    qformer_input_ids=torch.cat(qformer_input_ids_batch[batch_id:batch_end]).cuda(),
                    qformer_attention_mask=torch.cat(qformer_attention_mask_batch[batch_id:batch_end]).cuda(),
                    input_ids=torch.cat(input_ids_batch[batch_id:batch_end]).cuda(),
                    attention_mask=torch.cat(attention_mask_batch[batch_id:batch_end]).cuda(),
                    cache_image_embeds=True,
                    segmentation_mask=torch.cat(segmentation_masks_batch[batch_id:batch_end]).cuda(),
                    input_context_mask=torch.cat(context_mask_batch[batch_id:batch_end]).cuda(),
                    dataset_type="any",
                    max_new_tokens=16,
                    num_beams=1,
                    return_dict_in_generate=True,
                    output_scores=True).cpu() # cpu for sake of mem
            for j in range(generated_output_qformer.shape[0]):
                qformer_embed.append(generated_output_qformer[j])
            torch.cuda.empty_cache()


        return qformer_embed
    @torch.no_grad()
    def query_llm(self, representatives):
        len_ = representatives.shape[0]
        # batch initialization
        input_ids_batch = []
        attention_mask_batch = []
        
        for _ in range(len_):
            input_ids_batch.append(self.lang_x["input_ids"])
            attention_mask_batch.append(self.lang_x["attention_mask"])        
        
        generated_output = self.class_generator.generate_1n_avg_language(
                                qformer_output=representatives.cuda(),
                                input_ids=torch.cat(input_ids_batch[0:1]).cuda(),
                                attention_mask=torch.cat(attention_mask_batch[0:1]).cuda(),
                                dataset_type="any",
                                max_new_tokens=16,
                                num_beams=1,
                                return_dict_in_generate=True,
                                output_scores=True)
        generated_text = generated_output["sequences"][0]
        try:
            gentext = self.processor.tokenizer.decode(generated_text).split('</s>')[1].strip()
        except:
            gentext = 'None' # format of the sentence change due to avg
        return gentext

query_model_osm = None

class freevocab():
    def __init__(self, mask3d, mask2d, pcl, data2d, gtdata, pathdata2d, interval, osm_checkpoint, scene_id, pointdata, save_path = '../Dataset/QFormer_Feature_SAM_200/'):
        
        self.frames = [] ### Series of frames looping by interval step
        self.proposal3d = None ### np array of 3D class-agnostic proposals
        self.conf3d = None ### np array of 3D class-agnostic proposals confidence score
        self.model = query_model_osm
        self.scene_id = scene_id
        self.save_path = save_path
        #### Set up dataloader ####
        try:
            os.makedirs(save_path)
        except:
            pass

        try:
            loader = ScannetReader(root_path = data2d)
        except FileNotFoundError:
            print('>>> Error: Data not found. Did you download it?')
            exit()

        print("Number of frames to load: ", len(loader))

        #### Point2Image mapper ####
        img_dim = (640, 480)
        cut_num_pixel_boundary = 10 # do not use the features on the image boundary
        self.point2img_mapper = PointCloudToImageMapper(
            image_dim=img_dim, intrinsics = np.loadtxt(os.path.join(pathdata2d, 'intrinsic_depth.txt')),
            cut_bound=cut_num_pixel_boundary)
        self.point, self.color, self.sem_label, self.ins_label = torch.load(gtdata) # compressed point
        self.point, self.color, _ , _ = torch.load(pointdata) # raw point
        mask3d = torch.load(mask3d)
        self.proposal3d = torch.tensor(mask3d['ins'])
        self.conf3d = torch.tensor(mask3d['conf'])

        self.mask2d = mask2d

        #### Processing SAM ####
        mask_path = self.mask2d
        mask_out = torch.load(mask_path)


        for i in trange(0, len(loader),interval):
            frame = loader[i]
            frame_id = frame['frame_id']  # str
            depth = frame['depth']  # (h, w)
            image = frame['image']  # (h, w, 3), [0-255], uint8
            image_path = frame['image_path']  # str
            intrinsics = frame['intrinsics']  # (3,3) - camera intrinsics
            pose = frame['pose']  # (4,4) - camera pose
            pcd = frame['pcd']  # (n, 3) - backprojected point coordinates in world coordinate system! not camera coordinate system! backprojected and transformed to world coordinate system
            pcd_color = frame['color']  # (n,3) - color values for each point
            
            image_detic = cv2.resize(image,(depth.shape[1], depth.shape[0])) # DETIC image ~ depth shape
            #### Point mapping ####
            n_points = self.point.shape[0]
            mapping = np.ones([n_points, 4], dtype=int)
            mapping[:, 1:4] = self.point2img_mapper.compute_mapping(pose, self.point, depth)
            mapping = torch.tensor(mapping)
            # new_mapping = align_pixel(torch.squeeze(mapping[:, 1:3]), img_dim[1], img_dim[0], image.shape[0], image.shape[1])
            # mapping[:, 1:4] = torch.cat((new_mapping,mapping[:,3].unsqueeze(1)),dim=1)
            # indices of projectable pixels
            
            idx = torch.where(mapping[:,3] == 1)[0]
            # no points corresponds to this image, visible points on 2D image
            if mapping[:, 3].sum().item() == 0 or idx.shape[0]<100: 
                continue

            masks = []
            for mask in mask_out[frame_id]['masks']:
                try:
                    tmp = torch.tensor(pycocotools.mask.decode(mask))
                except:
                    continue
                masks.append(tmp)
            
            if len(masks) == 0:
                continue
            
            masks = torch.stack(masks)
            
            ### Filter small masks
            # Empty masks
            if masks == None:
                continue
            
            id = []
            for j in range(masks.shape[0]):
                if masks[j].sum()>=100:
                    id.append(j)
            if len(id)==0:
                continue

            masks = masks[id]

            #### Record Information ####
            dic = {}
            dic['image'] = image_detic
            dic['mapping'] = mapping
            dic['masks'] = masks
            self.frames.append(dic)
            #### Visualization of an image ####
            if False:
                # draw output image
                plt.figure(figsize=(10, 10))
                plt.imshow(image)
                for mask in masks:
                    show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
                plt.axis('off')
                plt.savefig(os.path.join("devfig/sam"+str(i)+'.jpg'), bbox_inches="tight", dpi=300, pad_inches=0.0)
                breakpoint()
    
    def align_majority(self, iou_threshold=0.0):
        ### Currently adopting maxIoU
        ### Returning indices of images and corresponding mask sets for each 3D proposals using IoU
        imgs = []
        highlight_masks = []   
        for mk3d in tqdm(self.proposal3d):
            img = []
            mask_id = []
            for i in range(len(self.frames)):
                frame = self.frames[i]
                npixels = frame['mapping'][torch.where(mk3d==1)[0]][:,3].sum().item() # related pixels
                if npixels < 5:
                    continue
                indices =  torch.where(frame['mapping'][torch.where(mk3d==1)[0]][:,3]==1)[0]
                sieve = torch.zeros_like(frame['masks'][0])
                sieve = sieve.to(torch.bool)
                r = frame['mapping'][torch.where(mk3d==1)[0]][indices][:,[1]]
                c = frame['mapping'][torch.where(mk3d==1)[0]][indices][:,[2]]

                expanded = sieve.unsqueeze(0).expand_as(frame['masks']).cuda()
                expanded[:, r, c] = True
                # Sieve masks with 2D masks
                mask = frame['masks'].cuda()
                logicAND = mask & expanded
                logicOR = mask | expanded
                intersection = torch.logical_and(logicAND, logicOR).sum(dim=(1, 2)).float()
                union = torch.logical_or(logicAND, logicOR).sum(dim=(1, 2)).float()
                iou = intersection / union # IoU tensor of given projected points on images

                idx = torch.where(iou>=0.0)[0] # Take every 2D mask of that view
                
                # if iou[idx] == 0.0:
                #     continue

                img.append(i)
                mask_id.append(idx)
                torch.cuda.empty_cache()
            # Record for each 3D instance mask
            imgs.append(img)
            highlight_masks.append(mask_id)
        self.imgs = imgs
        self.highlight_masks = highlight_masks

    def offline_images_batch(self):
        self.offline_query_class = []
        self.offline_query_prob = []        
        print('---OFFLINE PROCESSING---')
        image_set = []
        seg_masks=[]
        img_id = []
        interval = []
        cnt = 0
        for i in range(len(self.frames)):
            image_set.append(self.frames[i]['image'])
            interval.append(cnt)
            for mask in self.frames[i]['masks']:
                seg_masks.append(mask)
                img_id.append(i)
                cnt += 1
            interval.append(cnt)
        
        embeddings = self.model.process_multiple_image(image_set, seg_masks, img_id, interval, self.imgs, self.highlight_masks)
        
        # temporary offload to CPU for cuda ops
        self.model.class_generator.base_model = self.model.class_generator.base_model.cpu()
        torch.cuda.empty_cache()

        feat_bank = torch.zeros((self.point.shape[0], 32, 768), dtype=torch.float16).cpu() # QFormer Channel
        counter = torch.zeros((self.point.shape[0], 32, 768)).cpu()
        cnt = 0
        it = 0
        print('Aggregating features')
        for i in trange(len(self.frames)):
            frame = self.frames[i]
            mapping = frame['mapping']
            idx = torch.where(mapping[:,3] == 1)[0]
            start_frame = interval[it]
            end_frame = interval[it + 1]
            pred_masks = BitMasks(self.frames[i]['masks'])
            temp_mask = torch.nn.functional.interpolate(pred_masks.tensor.to(torch.float32).unsqueeze(1), size = (480,640), mode='bilinear', align_corners=False).squeeze(1)
            # batch_size = 10
            # # final_feat = torch.zeros((32,768, ))
            # mk_start = 0
            # for loop in range(start_frame, end_frame, batch_size):
            #     start_id = loop
            #     end_id = min(end_frame, loop + batch_size)
            #     mk_end = min(len(pred_masks), mk_start + batch_size)
            #     final_feat += torch.einsum("qcd,qhw->cdhw", torch.stack(embeddings[start_id:end_id]).to(torch.float16).cuda(), pred_masks[mk_start:mk_end].tensor.to(torch.float16).cuda()).cpu()
            #     mk_start += batch_size
            #     torch.cuda.empty_cache()
            batch_size = 100
            start_feat = 0
            end_feat = 768
            final_feat = torch.zeros((32,768,480,640))
            for loop in range(start_feat, end_feat, batch_size):
                start_id = loop
                end_id = min(end_feat, loop + batch_size)
                final_feat[:,start_id:end_id,:,:] += torch.einsum("qcd,qhw->cdhw", torch.stack(embeddings[start_frame:end_frame])[:,:,start_id:end_id].to(torch.float16).cuda(), temp_mask.to(torch.float16).cuda()).cpu()
                torch.cuda.empty_cache()
            # final_feat = torch.einsum("qcd,qhw->cdhw", torch.stack(embeddings[start_frame:end_frame]).to(torch.float16).cuda(), pred_masks.tensor.to(torch.float16).cuda()).cpu()

            jdx = torch.where(final_feat[0, 0, mapping[idx, 1], mapping[idx, 2]]!=0)[0].cpu()
            counter[idx[jdx]] += 1
            del jdx
            torch.cuda.empty_cache()
            
            feat_bank[idx] += final_feat[:, :, mapping[idx, 1], mapping[idx, 2]].permute(2,0,1)
            del final_feat
            torch.cuda.empty_cache()
            
            it += 2
            cnt += 1
        feat_bank[counter!=0]/=counter[counter!=0]
        torch.save({'feat':feat_bank}, self.save_path + self.scene_id + '.pth')
        torch.cuda.empty_cache()
        # GPU load for querying
        self.model.class_generator.base_model = self.model.class_generator.base_model.cuda()


    def query_offline(self):
        ### Query class name from pc feature offline
        pcfeature = torch.load(self.save_path + self.scene_id + '.pth')['feat']
        # pcfeature = F.normalize(pcfeature.to(torch.float32), dim=1, p=2)
        results = {'masks': [],'confidence2d': [], 'freq2d':[] ,'confidence3d': [], 'class': []}
        print('Processing per 3D proposals')
        for i in trange(len(self.imgs)):
            # Per 3D proposal
            if len(self.imgs[i]) == 0: # final_class[i] == None    
                continue
            cond1 = (self.proposal3d[i]==True)
            cond2 = (pcfeature[:,0,0] != 0)
            feat = pcfeature[cond1 & cond2].to(torch.float16)
            if feat.shape[0] == 0:
                continue
            pred_text = self.model.query_llm(feat)
            results['masks'].append(self.proposal3d[i])
            results['confidence3d'].append(self.conf3d[i])
            # class name querying from model
            results['class'].append(pred_text)
        torch.save(results, '../results_test_sam200/' + scene_id + '.pth')


if __name__=='__main__':
    
    # Load queryable model
    osm_checkpoint = '../weights/osm_final.pt'
    query_model_osm = OSM(osm_checkpoint)
    
    ######################### PerScene #########################
    per_scene = True
    if per_scene == True:
        #### datapath
        scene_id = 'scene0081_00'
        path3dmask = '../Dataset/Scannet200/class_ag_res_200_isbnetfull'
        path2dmask = '../Dataset/Scannet200/masksam_val'

        pathpcl = '../Dataset/ScannetV2/ScannetV2_2D_5interval/trainval/pcl' # Take XYZ, RGB (inst_no_stuff)
        pathdata2d = '../Dataset/ScannetV2/ScannetV2_2D_5interval/trainval' # scenes containing color/depth/pose
        pathgtdata = '../Dataset/Scannet200/val' # gt label of points following order compressed points
        pathpointdata = '../Dataset/ScannetV2/ScannetV2_2D_5interval/trainval/pcl'
        interval = 2
        
        #### merging path
        mask3d = os.path.join(path3dmask, scene_id + '.pth')
        mask2d = os.path.join(path2dmask, scene_id + '.pth')
        pcl = os.path.join(pathpcl, scene_id + '_inst_nostuff.pth')
        data2d = os.path.join(pathdata2d, scene_id)
        gtdata = os.path.join(pathgtdata, scene_id + '_inst_nostuff.pth')
        pointdata = os.path.join(pathpointdata, scene_id + '_inst_nostuff.pth')
        #### Processing
        container = freevocab(mask3d, mask2d, pcl, data2d, gtdata, pathdata2d, interval, osm_checkpoint, scene_id, pointdata)
        container.align_majority()
        container.offline_images_batch()
        container.query_offline()
    ###########################################################################
    if not per_scene:
        if os.path.exists('dataset3d/tracker_best.txt') == False:
            with open('dataset3d/tracker_best.txt', 'w') as file:
                file.write('Processed Scenes .\n')

        with open('dataset3d/scannet200_val.txt', 'r') as file:
            lines = file.readlines()
            lines = [line.strip() for line in lines]
        
        for scene_id in tqdm(lines):
            #### Check file existence
            done = False
            print('Processing ', scene_id)
            ## logging
            with open('dataset3d/tracker_best.txt', 'r') as file:
                tracks = file.readlines()
                tracks = [line.strip() for line in tracks]
                if scene_id in tracks:
                    done = True
            # directory
            file_names = os.listdir('../results_test')
            if (scene_id + '.pth') not in file_names:
                done = False

            if done == True:
                print('existed ', scene_id)
                continue
            with open('dataset3d/tracker_best.txt', 'a') as file:
                file.write(scene_id +'\n')
            #####    
            #### datapath
            path3dmask = '../Dataset/Scannet200/class_ag_res_200_isbnetfull'
            path2dmask = '../Dataset/ScannetV2/ScannetV2_detic_masks/imagenet21k-0.3'

            pathpcl = '../Dataset/ScannetV2/ScannetV2_2D_5interval/trainval/pcl' # Take XYZ, RGB (inst_no_stuff)
            pathdata2d = '../Dataset/ScannetV2/ScannetV2_2D_5interval/trainval' # scenes containing color/depth/pose
            pathgtdata = '../Dataset/Scannet200/val' # gt label of points following order compressed points
            pathpointdata = '../Dataset/ScannetV2/ScannetV2_2D_5interval/trainval/pcl'
            interval = 2
            
            #### merging path
            mask3d = os.path.join(path3dmask, scene_id + '.pth')
            mask2d = os.path.join(path2dmask, scene_id, 'instances')
            pcl = os.path.join(pathpcl, scene_id + '_inst_nostuff.pth')
            data2d = os.path.join(pathdata2d, scene_id)
            gtdata = os.path.join(pathgtdata, scene_id + '_inst_nostuff.pth')
            pointdata = os.path.join(pathpointdata, scene_id + '_inst_nostuff.pth')
            #### Processing
            container = freevocab(mask3d, mask2d, pcl, data2d, gtdata, pathdata2d, interval, osm_checkpoint, scene_id, pointdata)
            container.align_majority()
            container.offline_images_batch()
            container.query_offline()