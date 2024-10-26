import torch

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler, bbox_overlaps
from ..builder import HEADS, build_head, build_roi_extractor
from .base_roi_head import BaseRoIHead
from .test_mixins import BBoxTestMixin, MaskTestMixin

from mmdet.models.losses import accuracy
import math
import mmdet.core.bbox.iou_calculators.iou2d_calculator as iou

import copy

@HEADS.register_module()
class StandardRoIHeadDISCO(BaseRoIHead, BBoxTestMixin, MaskTestMixin):
    """Simplest base roi head including one bbox head and one mask head."""

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        self.bbox_assigner = None
        self.bbox_sampler = None
        if self.train_cfg:
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
            self.bbox_sampler = build_sampler(
                self.train_cfg.sampler, context=self)

    def init_bbox_head(self, bbox_roi_extractor, bbox_head):
        """Initialize ``bbox_head``"""
        self.bbox_roi_extractor = build_roi_extractor(bbox_roi_extractor)
        self.bbox_head = build_head(bbox_head)

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize ``mask_head``"""
        if mask_roi_extractor is not None:
            self.mask_roi_extractor = build_roi_extractor(mask_roi_extractor)
            self.share_roi_extractor = False
        else:
            self.share_roi_extractor = True
            self.mask_roi_extractor = self.bbox_roi_extractor
        self.mask_head = build_head(mask_head)

    def init_weights(self, pretrained):
        """Initialize the weights in head.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if self.with_shared_head:
            self.shared_head.init_weights(pretrained=pretrained)
        if self.with_bbox:
            self.bbox_roi_extractor.init_weights()
            self.bbox_head.init_weights()
        if self.with_mask:
            self.mask_head.init_weights()
            if not self.share_roi_extractor:
                self.mask_roi_extractor.init_weights()

    def forward_dummy(self, x, proposals):
        """Dummy forward function."""
        # bbox head
        outs = ()
        rois = bbox2roi([proposals])
        if self.with_bbox:
            bbox_results = self._bbox_forward(x, rois)
            outs = outs + (bbox_results['cls_score'],
                           bbox_results['bbox_pred'])
        # mask head
        if self.with_mask:
            mask_rois = rois[:100]
            mask_results = self._mask_forward(x, mask_rois)
            outs = outs + (mask_results['mask_pred'], )
        return outs

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """                
        
        def get_sampling_results(img_metas, proposal_list, gt_bboxes, gt_bboxes_ignore, gt_labels):
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)
            return sampling_results

        losses = dict()           

        corrected_gt_bboxes = gt_bboxes
        corrected_gt_labels = gt_labels
        assert self.bbox_head.assgin_time >= 1
        if not self.bbox_head.iter_assign_flag:
            self.bbox_head.assgin_time = 1

        for _ in range(self.bbox_head.assgin_time):
            # get sampling results of proposals
            if self.with_bbox or self.with_mask:
                sampling_results = get_sampling_results(img_metas, proposal_list, corrected_gt_bboxes, gt_bboxes_ignore, corrected_gt_labels)
            # FasterRCNN second stage forward
            bbox_results, rois, bbox_targets = self._bbox_forward_train(x, sampling_results, corrected_gt_bboxes, corrected_gt_labels, img_metas, **kwargs)
            # apply DISCO
            loss_disco, loss_bbox, corrected_gt_bboxes, corrected_gt_labels = self._distribution_aware_calibration(bbox_targets, bbox_results, rois, x)
            # break if warming up
            if corrected_gt_bboxes is None:
                break
        
        losses.update(loss_bbox)
        losses.update(loss_disco)

        return losses

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)

        cls_score, bbox_pred, bbox_var = self.bbox_head(bbox_feats)
        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_var=bbox_var, bbox_feats=bbox_feats)
        return bbox_results

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels, img_metas, **kwargs):
        """Run forward function and compute loss for box head during training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg, **kwargs)
        return bbox_results, rois, bbox_targets

    def _distribution_aware_calibration(self, bbox_targets, bbox_results, rois, x):

        loss_disco = dict()

        '''
        1. Distinguish positive proposal clusters
        '''
        labels, cur_bbox_targets = bbox_targets[0], bbox_targets[2]
        pos_inds = (labels >= 0) & (labels < self.bbox_head.num_classes)
        inds = torch.ones(pos_inds.sum()).cuda()
        pos_bbox_targets = cur_bbox_targets[pos_inds.type(torch.bool)]
        noisy_gt_boxes = self.bbox_head.bbox_coder.decode(rois[:, 1:][pos_inds.type(torch.bool)], pos_bbox_targets)
        uniq_inst, pos_indices = torch.unique(noisy_gt_boxes.sum(dim=1), sorted=True, return_inverse=True)
        assigned_proposal_indices = [torch.where(pos_indices == inst)[0] for inst in torch.unique(pos_indices)]

        '''
        2. get refined boxes and their confidence
        '''
        # get prediction of each proposal
        bbox_pred = bbox_results['bbox_pred'].view(bbox_results['bbox_pred'].shape[0], -1, 4)[pos_inds.type(torch.bool)]
        inds = torch.ones(bbox_pred.size(0)).type(torch.bool).cuda()
        pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1, 4)[inds, labels[pos_inds.type(torch.bool)]]
        # decode prediction of each proposal
        new_pred_boxes = self.bbox_head.bbox_coder.decode(rois[:, 1:][pos_inds.type(torch.bool)], pos_bbox_pred)
        new_roi = rois[pos_inds.type(torch.bool)].clone()
        new_roi[:,1:] = new_pred_boxes
        # get scores
        new_bbox_results = self._bbox_forward(x, new_roi)
        confidence_scores = torch.softmax(new_bbox_results['cls_score'], dim=1)[inds.type(torch.bool), labels[pos_inds.type(torch.bool)]]

        '''
        3. model distributions
        '''
        dist = {}
        dist_num = len(assigned_proposal_indices)
        dist['mean'] = torch.zeros((dist_num, 4)).cuda()
        dist['std'] = torch.zeros((dist_num, 4)).cuda()
        dist['gt'] = torch.zeros((dist_num, 4)).cuda()
        dist['id'] = torch.zeros(dist_num).cuda()
        dist['label'] = torch.zeros(dist_num).cuda()

        sampled_roi = torch.zeros((dist_num, self.bbox_head.sample_num, 5)).cuda()
        sampled_label = torch.zeros((dist_num, self.bbox_head.sample_num)).cuda()

        for idx, index in enumerate(assigned_proposal_indices):
            # get confidence score of each proposal
            assigned_proposal_boxes = new_pred_boxes[index]
            assigned_proposal_scores = confidence_scores[index]
            # calculate weights
            bbox_weights = torch.softmax(assigned_proposal_scores / self.bbox_head.temper_coef, dim=0).view(-1, 1)
            # calculate mean and std
            dist['mean'][idx] = torch.sum(bbox_weights * assigned_proposal_boxes, dim=0)
            dist['std'][idx] = torch.sqrt(torch.sum(bbox_weights * (assigned_proposal_boxes - dist['mean'][idx]) ** 2, dim=0))
            dist['gt'][idx] = noisy_gt_boxes[index[0]]
            dist['id'][idx] = rois[pos_inds.type(torch.bool)][index][0, 0]
            dist['label'][idx] = labels[pos_inds.type(torch.bool)][index][0]
            # sample & compare
            sampled_roi[idx, :, 0] = dist['id'][idx]
            sampled_roi[idx, :, 1:] = (dist['mean'][idx] + dist['std'][idx] * torch.randn(self.bbox_head.sample_num, 4).cuda()).clamp(min=0.0) 
            sampled_label[idx, :] = dist['label'][idx]           

        sampled_roi = sampled_roi.view(-1, 5)
        sampled_label = sampled_label.view(-1, )
        inds = torch.ones(sampled_roi.shape[0]).cuda()
        sampled_bbox_results = self._bbox_forward(x, sampled_roi)
        sampled_confidence_scores = torch.softmax(sampled_bbox_results['cls_score'], dim=1)[inds.type(torch.bool), sampled_label.long()]
        sampled_confidence_scores = sampled_confidence_scores.view(dist_num, self.bbox_head.sample_num)

        reg_calib_flag = self.bbox_head.reg_calib_flag and (self._epoch + 1 >= self.bbox_head.reg_calib_epoch)

        selected_confidence_scores = []
        sampled_pred_boxes = sampled_roi.view(dist_num, self.bbox_head.sample_num, 5)[:, :, 1:]
        pseudo_bbox_targets = torch.zeros_like(cur_bbox_targets[pos_inds.type(torch.bool)]).cuda() if reg_calib_flag else None
        pseudo_bbox_logvar = torch.zeros_like(cur_bbox_targets[pos_inds.type(torch.bool)]).cuda() if reg_calib_flag and self.bbox_head.variance_flag else None
        best_selected_bboxes = torch.zeros((dist_num, 4)).cuda() if reg_calib_flag else None

        phi_list = []
        original_cnt = 0

        for idx, index in enumerate(assigned_proposal_indices):
            '''
            4. select max scores
            '''
            # get confidence scores
            assigned_proposal_boxes = torch.cat([new_pred_boxes[index], sampled_pred_boxes[idx]], dim=0) 
            assigned_proposal_scores = torch.cat([confidence_scores[index], sampled_confidence_scores[idx]], dim=0) 
            if self.bbox_head.cls_calib_flag:
                # select
                topk_box_indces = torch.sort(assigned_proposal_scores, descending=True)[1][: self.bbox_head.selected_num]
                selected_confidence_scores.append(assigned_proposal_scores[topk_box_indces])
                # count
                orginal_box_num = new_pred_boxes[index].shape[0]
                original_cnt += torch.sum(topk_box_indces < orginal_box_num)
    
            '''
            5. refine distributions
            '''
            if self.bbox_head.dist_refine_flag:
                # calculate box weights
                bbox_weights = torch.softmax(assigned_proposal_scores / self.bbox_head.temper_coef, dim=0).view(-1, 1)
                # update mean and std
                dist['mean'][idx] = torch.sum(bbox_weights * assigned_proposal_boxes, dim=0)
                dist['std'][idx] = torch.sqrt(torch.sum(bbox_weights * (assigned_proposal_boxes - dist['mean'][idx]) ** 2, dim=0))

            '''
            6. get pseudo ground-truth
            '''
            if reg_calib_flag:
                # _, max_inds = torch.max(assigned_proposal_scores, dim=0)
                # best_score = assigned_proposal_scores[max_inds].clone()
                selected_bbox = dist['mean'][idx].clone()
                selected_bbox = selected_bbox.clamp(min=0.0).view(1, -1)
                
                mean_roi = torch.zeros((1, 5)).cuda()
                mean_roi[0, 0] = dist['id'][idx]
                mean_roi[0, 1:] = selected_bbox
                mean_label = dist['label'][idx]
                mean_bbox_results = self._bbox_forward(x, mean_roi)
                mean_score = torch.softmax(mean_bbox_results['cls_score'], dim=1)[0, mean_label.long()]

                # mix two bboxes
                phi = (mean_score.detach() ** self.bbox_head.mix_a).clamp(max=self.bbox_head.mix_b)
                phi_list.append(phi)
                best_selected_bbox = selected_bbox.detach() * phi + dist['gt'][idx].view(1, -1) * (1 - phi) 
                best_selected_bboxes[idx] = best_selected_bbox
                
                # reset grount-truth according to best selected bbox
                pseudo_gt_targets = self.bbox_head.bbox_coder.encode(rois[:, 1:][pos_inds.type(torch.bool)][index], best_selected_bbox.repeat(len(index), 1))
                pseudo_bbox_targets[index] = pseudo_gt_targets

                if self.bbox_head.variance_flag:
                    # pseudo_bbox_logvar[index] = torch.log((dist['std'][idx] ** 2).clamp(min=1e-6)).detach()
                    w = best_selected_bbox[0, 2] - best_selected_bbox[0, 0]
                    h = best_selected_bbox[0, 3] - best_selected_bbox[0, 1]
                    scaling_wh = torch.tensor([[w, h, w, h]]).cuda()
                    pseudo_bbox_logvar[index] = torch.log((dist['std'][idx] ** 2 / scaling_wh).clamp(min=1e-6)).detach()
                
        '''
        7. compute losses
        '''
        if self.bbox_head.cls_calib_flag:        
            selected_confidence_scores = torch.cat(selected_confidence_scores, dim=0)
            loss_disco['loss_cls_calib'] = -self.bbox_head.loss_lambda * selected_confidence_scores.log().mean()

        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'], bbox_results['bbox_pred'], bbox_results['bbox_var'], rois, *bbox_targets, pseudo_bbox_targets=pseudo_bbox_targets, pseudo_bbox_logvar=pseudo_bbox_logvar)

        '''
        8. get correct gt
        '''
        if reg_calib_flag:  
            # sorted_id_indices = [torch.where(dist['id'] == id)[0] for id in torch.unique(dist['id'])]
            sorted_id_indices = [torch.where(dist['id'] == id)[0] for id in range(self.bbox_head.samples_per_gpu)]
            corrected_gt_bboxes = [best_selected_bboxes[index] for index in sorted_id_indices]
            corrected_gt_labels = [dist['label'][index].long() for index in sorted_id_indices]
        else:
            corrected_gt_bboxes = None
            corrected_gt_labels = None

        loss_disco['dist_var'] = (dist['std'] ** 2).mean()
        w = (dist['mean'][:, 2] - dist['mean'][:, 0]).view(-1, 1)
        h = (dist['mean'][:, 3] - dist['mean'][:, 1]).view(-1, 1)
        scaling_wh = torch.cat([w, h, w, h], dim=1).clamp(min=1e-6)
        loss_disco['dist_var_scaled'] = (dist['std'] ** 2 / scaling_wh).mean()
        
        loss_disco['dist_iou'] = (bbox_overlaps(dist['mean'], dist['gt'], is_aligned=True).clamp(min=1e-6)).mean()

        if len(phi_list):
            loss_disco['phi'] = sum(phi_list) / len(phi_list)

        if reg_calib_flag and self.bbox_head.variance_flag:
            pos_bbox_var = bbox_results['bbox_var'][pos_inds.type(torch.bool)]
            inds = torch.ones(pos_bbox_var.size(0)).type(torch.bool).cuda()
            pos_bbox_var = pos_bbox_var.view(pos_bbox_var.size(0), -1, 4)[inds, labels[pos_inds.type(torch.bool)]]
            pos_bbox_var = torch.exp(pos_bbox_var)
            loss_disco['bbox_var_scaled'] = pos_bbox_var.mean()
            w = (best_selected_bboxes[:, 2] - best_selected_bboxes[:, 0]).view(-1, 1)
            h = (best_selected_bboxes[:, 3] - best_selected_bboxes[:, 1]).view(-1, 1)
            scaling_wh = torch.cat([w, h, w, h], dim=1).cuda()
            scaling_wh_bbox = torch.zeros((pos_bbox_var.shape[0], 4)).cuda()
            for idx, index in enumerate(assigned_proposal_indices):
                scaling_wh_bbox[index] = scaling_wh[idx]
            pos_bbox_var *= scaling_wh_bbox
            loss_disco['bbox_var'] =  pos_bbox_var.mean()

        return loss_disco, loss_bbox, corrected_gt_bboxes, corrected_gt_labels
    
    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            if pos_rois.shape[0] == 0:
                return dict(loss_mask=None)
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)
            if pos_inds.shape[0] == 0:
                return dict(loss_mask=None)
            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        if self.test_cfg is None:
            det_bboxes, det_labels = self.simple_test_bboxes(
                x, img_metas, proposal_list, None, rescale=rescale)
            det_bboxes = [boxes.cpu().numpy() for boxes in det_bboxes]
            det_labels = [labels.cpu().numpy() for labels in det_labels]
            return det_bboxes, det_labels

        else:
            det_bboxes, det_labels = self.simple_test_bboxes(
                x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
            bbox_results = [
                bbox2result(det_bboxes[i], det_labels[i],
                            self.bbox_head.num_classes)
                for i in range(len(det_bboxes))
            ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

    def aug_test(self, x, proposal_list, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        det_bboxes, det_labels = self.aug_test_bboxes(x, img_metas,
                                                      proposal_list,
                                                      self.test_cfg)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= det_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(x, img_metas, det_bboxes,
                                              det_labels)
            return [(bbox_results, segm_results)]
        else:
            return [bbox_results]

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):                    
        """Test only det bboxes without augmentation."""
        rois = bbox2roi(proposals)
        bbox_results = self._bbox_forward(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)        

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']

        if self.bbox_head.var_vote:
            bbox_var = bbox_results['bbox_var']
        else:
            bbox_var = None

        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        # some detector with_reg is False, bbox_pred will be None
        bbox_pred = bbox_pred.split(
            num_proposals_per_img,
            0) if bbox_pred is not None else [None, None]

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            det_bbox, det_label = self.bbox_head.get_bboxes(
                rois[i],
                cls_score[i],
                bbox_pred[i],
                img_shapes[i],
                scale_factors[i],
                rescale=rescale,
                cfg=rcnn_test_cfg,
                variance=bbox_var)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels