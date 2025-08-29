"""Knowledge distillation module for model compression."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any
import warnings


class DistillationLoss(nn.Module):
    """Knowledge distillation loss for segmentation models.
    
    Combines standard segmentation loss with knowledge distillation from teacher model.
    """
    
    def __init__(
        self,
        base_loss_fn: nn.Module,
        temperature: float = 4.0,
        alpha: float = 0.7,
        distill_logits: bool = True,
        distill_features: bool = False,
        feature_match_layers: Optional[list] = None
    ):
        """Initialize distillation loss.
        
        Args:
            base_loss_fn: Base segmentation loss function (e.g., HierarchicalLoss)
            temperature: Temperature for softening probability distributions
            alpha: Weight for distillation loss (1-alpha for base loss)
            distill_logits: Whether to distill output logits
            distill_features: Whether to distill intermediate features
            feature_match_layers: List of layer names for feature matching
        """
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.temperature = temperature
        self.alpha = alpha
        self.distill_logits = distill_logits
        self.distill_features = distill_features
        self.feature_match_layers = feature_match_layers or []
        
        # KL divergence loss for soft targets
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        
        # MSE loss for feature matching
        self.mse_loss = nn.MSELoss()
        
    def forward(
        self, 
        student_outputs: Any,
        teacher_outputs: Any,
        targets: torch.Tensor,
        aux_outputs: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined distillation and base loss.
        
        Args:
            student_outputs: Student model outputs (logits or tuple of logits and aux)
            teacher_outputs: Teacher model outputs (logits or tuple of logits and aux) 
            targets: Ground truth masks
            aux_outputs: Auxiliary outputs (for compatibility with hierarchical models)
            
        Returns:
            loss: Combined loss
            loss_dict: Dictionary of individual loss components
        """
        loss_dict = {}
        
        # Handle hierarchical model outputs (tuple of logits and aux_outputs)
        if isinstance(student_outputs, tuple):
            student_logits, student_aux = student_outputs
        else:
            student_logits = student_outputs
            student_aux = aux_outputs
            
        if isinstance(teacher_outputs, tuple):
            teacher_logits, teacher_aux = teacher_outputs
        else:
            teacher_logits = teacher_outputs
            teacher_aux = None
        
        
        # 1. Base segmentation loss (student vs ground truth)
        if student_aux is not None:
            base_loss, base_loss_dict = self.base_loss_fn(student_logits, targets, student_aux)
        else:
            base_loss, base_loss_dict = self.base_loss_fn(student_logits, targets)
        
        # Add base loss components to loss dict
        for key, value in base_loss_dict.items():
            loss_dict[f'base_{key}'] = value
        
        # 2. Distillation loss (student vs teacher)
        distill_loss = 0
        
        if self.distill_logits:
            # Logit distillation with temperature scaling
            student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
            teacher_soft = F.softmax(teacher_logits.detach() / self.temperature, dim=1)
            
            # KL divergence weighted by temperature squared (maintains gradient magnitudes)
            logit_distill_loss = self.kl_loss(student_soft, teacher_soft) * (self.temperature ** 2)
            distill_loss += logit_distill_loss
            loss_dict['distill_logits'] = logit_distill_loss.item()
            
        # 3. Auxiliary output distillation (if applicable)
        if student_aux is not None and teacher_aux is not None:
            # Distill foreground/background predictions
            if 'bg_fg_logits' in student_aux and 'bg_fg_logits' in teacher_aux:
                student_fg_soft = F.log_softmax(student_aux['bg_fg_logits'] / self.temperature, dim=1)
                teacher_fg_soft = F.softmax(teacher_aux['bg_fg_logits'].detach() / self.temperature, dim=1)
                
                fg_distill_loss = self.kl_loss(student_fg_soft, teacher_fg_soft) * (self.temperature ** 2) * 0.3
                distill_loss += fg_distill_loss
                loss_dict['distill_bg_fg'] = fg_distill_loss.item()
                
            # Distill target/non-target predictions if available
            if 'target_nontarget_logits' in student_aux and 'target_nontarget_logits' in teacher_aux:
                student_target_soft = F.log_softmax(student_aux['target_nontarget_logits'] / self.temperature, dim=1)
                teacher_target_soft = F.softmax(teacher_aux['target_nontarget_logits'].detach() / self.temperature, dim=1)
                
                target_distill_loss = self.kl_loss(student_target_soft, teacher_target_soft) * (self.temperature ** 2) * 0.3
                distill_loss += target_distill_loss
                loss_dict['distill_target'] = target_distill_loss.item()
        
        # Combine losses with alpha weighting
        total_loss = self.alpha * distill_loss + (1 - self.alpha) * base_loss
        
        loss_dict['distill_total'] = distill_loss.item() if isinstance(distill_loss, torch.Tensor) else distill_loss
        loss_dict['base_total'] = base_loss.item()
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


class DistillationModelWrapper(nn.Module):
    """Wrapper for teacher-student distillation training.
    
    This wrapper manages both teacher and student models during training.
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        freeze_teacher: bool = True
    ):
        """Initialize distillation wrapper.
        
        Args:
            teacher_model: Pre-trained teacher model
            student_model: Student model to be trained
            freeze_teacher: Whether to freeze teacher model weights
        """
        super().__init__()
        self.teacher = teacher_model
        self.student = student_model
        
        # Freeze teacher model if requested
        if freeze_teacher:
            for param in self.teacher.parameters():
                param.requires_grad = False
            self.teacher.eval()  # Set to eval mode
            
    def forward(self, *args, **kwargs) -> Tuple[Any, Any]:
        """Forward pass through both teacher and student models.
        
        Returns:
            student_outputs: Student model outputs
            teacher_outputs: Teacher model outputs (detached if frozen)
        """
        # Get student outputs (with gradients)
        student_outputs = self.student(*args, **kwargs)
        
        # Get teacher outputs (without gradients if frozen)
        if not self.teacher.training:
            with torch.no_grad():
                teacher_outputs = self.teacher(*args, **kwargs)
        else:
            teacher_outputs = self.teacher(*args, **kwargs)
            
        return student_outputs, teacher_outputs
    
    def train(self, mode: bool = True):
        """Set training mode for student only (teacher stays in eval if frozen)."""
        self.student.train(mode)
        # Teacher stays in eval mode if frozen
        if not any(p.requires_grad for p in self.teacher.parameters()):
            self.teacher.eval()
        else:
            self.teacher.train(mode)
        return self
    
    def eval(self):
        """Set both models to eval mode."""
        self.student.eval()
        self.teacher.eval()
        return self
    
    def get_student(self) -> nn.Module:
        """Get the student model for saving/exporting."""
        return self.student
    
    def get_teacher(self) -> nn.Module:
        """Get the teacher model."""
        return self.teacher