# Filename: main.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import RTDETR, YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
import os

# Global storage for features, made process-safe
teacher_features = {}
student_features = {}

def get_features_hook(storage, name):
    def hook(model, input, output):
        storage[os.getpid()] = {name: output}
    return hook


class YOLOKDTrainer(DetectionTrainer):
    """
    Custom YOLO trainer for distillation.
    """
    def _setup_train(self, world_size=1):
        super()._setup_train()

        self.teacher = RTDETR('rtdetr-x.pt').to(self.device)
        self.teacher.eval()
        print("RT-DETR-x teacher model loaded and in evaluation mode.")

        teacher_layer = self.teacher.model.model[1]
        teacher_layer.register_forward_hook(get_features_hook(teacher_features, 'distill'))

        student_layer = self.model.model[9]
        student_layer.register_forward_hook(get_features_hook(student_features, 'distill'))

        self.adapter = nn.Conv2d(512, 1024, kernel_size=1).to(self.device)
        print("Trainer hooks and adapter layer initialized successfully.")

        # THE FINAL FIX: Add the missing 'initial_lr' key for the LR scheduler.
        self.optimizer.add_param_group({
            'params': self.adapter.parameters(),
            'initial_lr': self.args.lr0  # This is the crucial missing piece
        })

    def criterion(self, preds, batch):
        loss_detect, loss_items_detect = super().criterion(preds, batch)

        with torch.no_grad():
            _ = self.teacher(batch['img'])
        
        pid = os.getpid()
        
        if pid in teacher_features and pid in student_features:
            t_feats_list = teacher_features[pid]['distill']
            t_feats = t_feats_list[2]

            s_feats = student_features[pid]['distill']
            s_feats_adapted = self.adapter(s_feats)
            
            s_feats_adapted = F.interpolate(s_feats_adapted, size=t_feats.shape[-2:], mode='bilinear', align_corners=False)

            loss_distill_feat = F.mse_loss(s_feats_adapted, t_feats)
            distill_weight = 10.0

            total_loss = loss_detect + (distill_weight * loss_distill_feat)
            loss_items_distill = torch.cat((loss_items_detect, (distill_weight * loss_distill_feat).unsqueeze(0)))

            if pid in teacher_features: del teacher_features[pid]
            if pid in student_features: del student_features[pid]

            return total_loss, loss_items_distill
        
        return loss_detect, loss_items_detect


if __name__ == '__main__':
    student_model = YOLO('yolov8n.pt')
    dataset_yaml = 'distil_dataset.yaml'

    if not os.path.exists(dataset_yaml):
        print(f"FATAL: Dataset configuration '{dataset_yaml}' not found.")
    else:
        student_model.train(
            data=dataset_yaml,
            trainer=YOLOKDTrainer,
            epochs=100,
            batch=16,
            imgsz=640,
            workers=0
        )
        print("Distillation training complete. Best model saved in 'runs/detect/train'.")
