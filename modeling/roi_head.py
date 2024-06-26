from pydoc import classname
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torchvision.ops as ops
import pickle
import torchvision.transforms as T


from detectron2.layers import ShapeSpec
from detectron2.data import MetadataCatalog

from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY,  Res5ROIHeads
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from .box_predictor import ClipFastRCNNOutputLayers
import clip
import cv2
import xml.etree.ElementTree as ET
id = 0

mapp = {0 : "0c774f7d-a8305951",1 : "4f4592c1-6dc0f65d", 2: "5b5ec103-739c42df",3 : "fe172415-3c36f3d1"}




rois1 = []
labels1 = []
device = "cuda:3" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("RN101", device=device)
cnt = 3000
classes = {0 : 'car',1 : 'bike',2 : 'motor',3 : 'person', 4 : 'rider',5 : 'truck'}
reversed_classes = {v: k for k, v in classes.items()}

count = {'car' : cnt,'truck' : cnt,'rider' : cnt, 'person' : cnt, 'bike' : cnt,'motor' : cnt}
rois = []
labels = []
def select_foreground_proposals(
    proposals: List[Instances], bg_label: int
) -> Tuple[List[Instances], List[torch.Tensor]]:
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.
    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.
    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks


@ROI_HEADS_REGISTRY.register()
class ClipRes5ROIHeads(Res5ROIHeads):   
    def __init__(self, cfg, input_shape) -> None:
        super().__init__(cfg, input_shape)
        clsnames = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get("thing_classes").copy()

        # import pdb;pdb.set_trace()
        ##change the labels to represent the objects correctly
        for name in  cfg.MODEL.RENAME:
            ind = clsnames.index(name[0])
            clsnames[ind] = name[1]
       
        out_channels=cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * (2 ** 3) ### copied 
        self.box_predictor = ClipFastRCNNOutputLayers(cfg, ShapeSpec(channels=out_channels, height=1, width=1), clsnames)
        self.clip_im_predictor = self.box_predictor.cls_score # should call it properly
        self.device = cfg.MODEL.DEVICE
    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        crops: Optional[List[Tuple]] = None,
    ):
        """
        See :meth:`ROIHeads.forward`.
        """
        del images

        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
            # import pdb;pdb.set_trace()
            loss_crop_im = None
            if crops is not None:
                crop_im = list()#[x[0] for x in crops] #bxcropx3x224x224
                crop_boxes = list()#[x[1].to(self.device) for x in crops] #bxcropsx4
                keep = torch.ones(len(crops)).bool()
                
                for ind,x in enumerate(crops):
                    if len(x) == 0:
                        keep[ind] = False   
                        continue
                    crop_im.append(x[0])
                    crop_boxes.append(x[1].to(self.device))
                    
                c = self._shared_roi_transform(
                            [features[f][keep] for f in self.in_features], crop_boxes) #(b*crops)x2048x7x7
                loss_crop_im, _ = self.clip_im_predictor.forward_crops(crop_im,crops_features.mean(dim=[2, 3]))

        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        # nms_thresh = 0.5
        # final_proposals = []
        # for proposal in proposals:
        #     boxes = proposal.proposal_boxes.tensor
        #     scores = proposal.objectness_logits
        #     keep = ops.nms(boxes, scores, nms_thresh)
        #     final_proposals.append(proposal[keep])
        # with open('final_proposals_first.pkl', 'wb') as file:
        #     pickle.dump(final_proposals, file)
        predictions = self.box_predictor(box_features.mean(dim=[2, 3]))
        # import pdb;pdb.set_trace()
        if self.training:
            del features
            losses = self.box_predictor.losses(predictions, proposals)
            if self.mask_on:
                proposals, fg_selection_masks = select_foreground_proposals(
                    proposals, self.num_classes
                )
                # Since the ROI feature transform is shared between boxes and masks,
                # we don't need to recompute features. The mask loss is only defined
                # on foreground proposals, so we need to select out the foreground
                # features.
                mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]
                del box_features
                losses.update(self.mask_head(mask_features, proposals))

            if loss_crop_im is not None:
                losses.update(loss_crop_im)
            return [], losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}


@ROI_HEADS_REGISTRY.register()
class ClipRes5ROIHeadsAttn(ClipRes5ROIHeads): 
    def __init__(self, cfg, input_shape) -> None:
        super().__init__(cfg, input_shape)
        # self.res5 = None
    
    def _shared_roi_transform(self, features, boxes):
        x = self.pooler(features, boxes)
        return self.fwdres5(x)

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        crops: Optional[List[Tuple]] = None,
        backbone = None
    ):
        """
        See :meth:`ROIHeads.forward`.
        """
        del images

        self.fwdres5 = backbone.forward_res5

        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
            # import pdb;pdb.set_trace()
            loss_crop_im = None
            if crops is not None:
                crop_im = list()#[x[0] for x in crops] #bxcropx3x224x224
                crop_boxes = list()#[x[1].to(self.device) for x in crops] #bxcropsx4
                keep = torch.ones(len(crops)).bool()
                
                for ind,x in enumerate(crops):
                    if len(x) == 0:
                        keep[ind] = False   
                        continue
                    crop_im.append(x[0])
                    crop_boxes.append(x[1].to(self.device))
                    
                crops_features = self._shared_roi_transform(
                            [features[f][keep] for f in self.in_features], crop_boxes) #(b*crops)x2048x7x7
                crops_features = backbone.attention_global_pool(crops_features)
                loss_crop_im, _ = self.clip_im_predictor.forward_crops(crop_im,crops_features)

        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        attn_feat = backbone.attention_global_pool(box_features)
        predictions = self.box_predictor([attn_feat,box_features.mean(dim=(2,3))])
        # import pdb;pdb.set_trace()
        if self.training:
            del features
            
            losses = self.box_predictor.losses(predictions, proposals)
            # if torch.isnan(losses['loss_cls']):
            #     import pdb;pdb.set_trace()
           
            if self.mask_on:
                proposals, fg_selection_masks = select_foreground_proposals(
                    proposals, self.num_classes
                )
                # Since the ROI feature transform is shared between boxes and masks,
                # we don't need to recompute features. The mask loss is only defined
                # on foreground proposals, so we need to select out the foreground
                # features.
                mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]
                del box_features
                losses.update(self.mask_head(mask_features, proposals))

            if loss_crop_im is not None:
                losses.update(loss_crop_im)
            return [], losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            # boxes = pred_instances[0]._fields['pred_boxes'].tensor.cpu().numpy()
            # global id
            # name = mapp[id]
            # id += 1
            # image_path = f"/u/student/2022/cs22mtech14005/Thesis1/zs2/domaingen/data/datasets/diverseWeather/daytime_clear/VOC2007/JPEGImages/{name}.jpg"
            # image = cv2.imread(image_path)
            # image = cv2.resize(image,(1067,600))     
            # i = 0 
            # for box in boxes:
            #     if pred_instances[0]._fields['pred_classes'][i].item() == 0 and pred_instances[0]._fields['scores'][i].item() >= 0.5:
            #         cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            #     i += 1
            # cv2.imwrite(f'{name}_rslt.jpg',image)
            # for x in range(_[0].__len__()):
            #         if rois.__len__() >= 6*cnt:
            #             break
            #         if pred_instances[0]._fields['scores'].detach().cpu()[x] >= 0.5 and count[classes[pred_instances[0]._fields['pred_classes'].detach().cpu()[x].item()]] >= 1:
            #             rois.append(box_features[_[0][x]].detach().cpu().numpy())
            #             labels.append(classes[pred_instances[0]._fields['pred_classes'].detach().cpu()[x].item()])
            #             count[classes[pred_instances[0]._fields['pred_classes'].detach().cpu()[x].item()]] -= 1
            # import numpy as np
            # global id
            # boxes = pred_instances[0]._fields['pred_boxes'].tensor.cpu().numpy()
            # pred_classes = pred_instances[0]._fields['pred_classes'].cpu().numpy()
            # pred_classes = pred_classes.reshape(pred_classes.__len__(),1)
            # boxes = np.hstack((boxes,pred_classes))
            # xml_file = "/u/student/2022/cs22mtech14005/Thesis1/zs2/domaingen/data/datasets/diverseWeather/daytime_clear/VOC2007/Annotations/" + map[f'{id}'] + '.xml'
            # ground_truth_boxes = parse_voc_xml_with_classes(xml_file)
            # valid_predictions = filter_predictions(boxes, ground_truth_boxes)
            # # print(valid_predictions)
            # id += 1
            # box_features = box_features[_[0]]
            # print(pred_instances)
            # for x in valid_predictions:
            #     rois.append(box_features[x].detach().cpu().numpy())
            #     labels.append(classes[pred_instances[0]._fields['pred_classes'].detach().cpu()[x].item()])
            #     count[classes[pred_instances[0]._fields['pred_classes'].detach().cpu()[x].item()]]
            return pred_instances, {},box_features[_[0]]



