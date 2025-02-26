import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from kornia.losses import ssim_loss

class PerfusionMapLoss(nn.Module):
    """
    Loss function comparing clinically relevant perfusion maps between 
    predicted and ground truth 4D scan data.
    
    Focuses on hemodynamic parameters most important for clinical 
    diagnosis while handling noisy data efficiently.
    """
    def __init__(self, time_values=None, cbf_weight=1.0, tmax_weight=1.0, 
                 cbv_weight=0.7, mtt_weight=0.5, struct_weight=0.3,
                 clinical_thresholds=True, batch_deconvolution=True):
        """
        Parameters:
        ----------
        time_values : torch.Tensor, optional
            Time points of the dynamic scan
        cbf_weight : float
            Weight for CBF comparison
        tmax_weight : float
            Weight for Tmax comparison
        cbv_weight : float
            Weight for CBV comparison  
        mtt_weight : float
            Weight for MTT comparison
        struct_weight : float
            Weight for structural similarity component
        clinical_thresholds : bool
            Whether to include clinical threshold-based loss components
        batch_deconvolution : bool
            Whether to use batched SVD deconvolution (more efficient)
        """
        super(PerfusionMapLoss, self).__init__()
        
        self.cbf_weight = cbf_weight
        self.tmax_weight = tmax_weight
        self.cbv_weight = cbv_weight
        self.mtt_weight = mtt_weight
        self.struct_weight = struct_weight
        self.clinical_thresholds = clinical_thresholds
        self.batch_deconvolution = batch_deconvolution
        
        # Use provided time values or default
        if time_values is None:
            self.register_buffer('time_values', torch.arange(0, 60, dtype=torch.float32))
        else:
            self.register_buffer('time_values', time_values)
            
        # SVD regularization parameter
        self.svd_reg = 0.2
        
        # Clinical thresholds for different parameters
        self.register_buffer('cbf_thresh', torch.tensor([10.0, 20.0, 50.0]))  # mL/100g/min
        self.register_buffer('tmax_thresh', torch.tensor([6.0, 10.0]))        # seconds
        self.register_buffer('cbv_thresh', torch.tensor([2.0, 4.0]))          # mL/100g
        
        # Weight maps initialization
        self.tissue_weight_map = None
    
    def _extract_aif(self, volume, batch_idx=0):
        """
        Extract arterial input function from brightest, earliest enhancing voxels
        
        Parameters:
        ----------
        volume : torch.Tensor
            4D tensor of shape [B, T, H, W] or [B, T, D, H, W]
        batch_idx : int
            Batch index to use for AIF extraction
            
        Returns:
        -------
        torch.Tensor
            AIF curve
        """
        if len(volume.shape) == 4:  # [B, T, H, W]
            temp_vol = volume[batch_idx]  # [T, H, W]
            # Find time of maximum for each voxel
            arrival_time = torch.argmax(temp_vol, dim=0)
            # Find maximum value for each voxel
            max_val = torch.max(temp_vol, dim=0).values
            
            # Create mask for early arriving, high intensity voxels
            mask = (arrival_time < temp_vol.shape[0] * 0.3) & (max_val > torch.quantile(max_val, 0.95))
            
            if mask.sum() > 0:
                # Get mean curve of the masked voxels
                masked_voxels = temp_vol[:, mask]
                aif = torch.mean(masked_voxels, dim=1)
            else:
                # Fallback to central region
                h, w = temp_vol.shape[1:3]
                center_slice = temp_vol[:, h//2-2:h//2+2, w//2-2:w//2+2]
                aif = torch.mean(center_slice, dim=(1, 2))
                
        else:  # [B, T, D, H, W]
            temp_vol = volume[batch_idx]  # [T, D, H, W]
            # Similar approach but with 3D volume
            arrival_time = torch.argmax(temp_vol, dim=0) 
            max_val = torch.max(temp_vol, dim=0).values
            
            mask = (arrival_time < temp_vol.shape[0] * 0.3) & (max_val > torch.quantile(max_val, 0.95))
            
            if mask.sum() > 0:
                masked_voxels = temp_vol[:, mask]
                aif = torch.mean(masked_voxels, dim=1)
            else:
                d, h, w = temp_vol.shape[1:4]
                center_slice = temp_vol[:, d//2, h//2-2:h//2+2, w//2-2:w//2+2]
                aif = torch.mean(center_slice, dim=(1, 2))
            
        # Basic preprocessing
        baseline = torch.mean(aif[:max(3, int(aif.shape[0] * 0.1))])
        aif = F.relu(aif - baseline)  # Remove baseline and keep positive values
        
        # Area normalization
        dt = self.time_values[1] - self.time_values[0] if self.time_values.shape[0] > 1 else 1.0
        area = torch.sum(aif) * dt
        if area > 0:
            aif = aif / area
            
        return aif
    
    def _compute_perfusion_maps(self, volume):
        """
        Compute perfusion maps from 4D volume data using differentiable operations
        
        Parameters:
        ----------
        volume : torch.Tensor
            5D tensor of shape [B, T, C, H, W] or [B, T, C, D, H, W]
            
        Returns:
        -------
        dict
            Dictionary of perfusion maps
        """
        batch_size = volume.shape[0]
        n_timepoints = volume.shape[1]
        
        is_3d = len(volume.shape) == 6
        
        # Reshape to handle both 2D+time and 3D+time
        if is_3d:
            # [B, T, C, D, H, W] -> process each batch
            b, t, c, d, h, w = volume.shape
            # For simplicity, we'll just process the center slice
            volume_center = volume[:, :, :, d//2, :, :]  # [B, T, C, H, W]
        else:
            volume_center = volume
                
        # Initialize perfusion maps for the batch
        if is_3d:
            cbf_map = torch.zeros((batch_size, 1, d, h, w), device=volume.device)
            cbv_map = torch.zeros((batch_size, 1, d, h, w), device=volume.device)
            mtt_map = torch.zeros((batch_size, 1, d, h, w), device=volume.device)
            delay_map = torch.zeros((batch_size, 1, d, h, w), device=volume.device)
            tmax_map = torch.zeros((batch_size, 1, d, h, w), device=volume.device)
        else:
            cbf_map = torch.zeros((batch_size, 1, h, w), device=volume.device)
            cbv_map = torch.zeros((batch_size, 1, h, w), device=volume.device)
            mtt_map = torch.zeros((batch_size, 1, h, w), device=volume.device)
            delay_map = torch.zeros((batch_size, 1, h, w), device=volume.device)
            tmax_map = torch.zeros((batch_size, 1, h, w), device=volume.device)
        
        # Process each sample in batch
        for b_idx in range(batch_size):
            # Extract AIF (using first channel if multi-channel)
            aif = self._extract_aif(volume[:, :, 0] if volume.shape[2] > 1 else volume, b_idx)
            
            # Prepare AIF for deconvolution (create Toeplitz-like matrix)
            aif_matrix = torch.zeros(n_timepoints, n_timepoints, device=volume.device)
            for i in range(n_timepoints):
                aif_matrix[i, :i+1] = aif[:i+1]
            
            # Efficient SVD deconvolution
            try:
                # Compute SVD
                U, S, V = torch.svd(aif_matrix)
                
                # Truncate small singular values for stability
                threshold = self.svd_reg * torch.max(S)
                S_inv = torch.zeros_like(S)
                S_inv[S > threshold] = 1.0 / S[S > threshold]
                
                # Create diagonal matrix
                S_inv_mat = torch.diag(S_inv)
                
                # Compute inverse (for deconvolution)
                inv_aif_matrix = torch.matmul(torch.matmul(V, S_inv_mat), U.transpose(-2, -1))
                
                # Get proper slice for processing
                if is_3d:
                    vol_slice = volume[b_idx, :, 0]  # [T, D, H, W]
                    
                    # Efficient computation with vectorized operations
                    # Reshape for matrix operations
                    t, d, h, w = vol_slice.shape
                    vol_flat = vol_slice.reshape(t, -1)  # [T, D*H*W]
                    
                    # Baseline correction
                    baseline = torch.mean(vol_flat[:max(3, int(t * 0.1))], dim=0)
                    vol_corr = F.relu(vol_flat - baseline)
                    
                    # Deconvolution for all voxels
                    residue_funcs = torch.matmul(inv_aif_matrix, vol_corr)  # [T, D*H*W]
                    residue_funcs = F.relu(residue_funcs)  # Ensure non-negativity
                    
                    # CBF = max of residue function
                    cbf_flat = torch.max(residue_funcs, dim=0).values
                    
                    # CBV = area under curve
                    dt = self.time_values[1] - self.time_values[0] if self.time_values.shape[0] > 1 else 1.0
                    cbv_flat = torch.sum(vol_corr, dim=0) * dt
                    
                    # Reshape back
                    cbf_map[b_idx, 0] = cbf_flat.reshape(d, h, w)
                    cbv_map[b_idx, 0] = cbv_flat.reshape(d, h, w)
                    
                    # MTT = CBV/CBF
                    valid_mask = cbf_flat > 0
                    mtt_flat = torch.zeros_like(cbf_flat)
                    mtt_flat[valid_mask] = cbv_flat[valid_mask] / cbf_flat[valid_mask]
                    mtt_map[b_idx, 0] = mtt_flat.reshape(d, h, w)
                    
                    # Delay = time to peak
                    peak_idx = torch.argmax(vol_flat, dim=0)
                    delay_map[b_idx, 0] = (self.time_values[peak_idx]).reshape(d, h, w)
                    
                    # Tmax = time to max of residue function
                    tmax_idx = torch.argmax(residue_funcs, dim=0)
                    tmax_map[b_idx, 0] = (self.time_values[tmax_idx]).reshape(d, h, w)
                
                else:
                    vol_slice = volume[b_idx, :, 0] if volume.shape[2] > 0 else volume[b_idx]  # [T, H, W]
                    
                    # Reshape for matrix operations
                    t, h, w = vol_slice.shape
                    vol_flat = vol_slice.reshape(t, -1)  # [T, H*W]
                    
                    # Baseline correction
                    baseline = torch.mean(vol_flat[:max(3, int(t * 0.1))], dim=0)
                    vol_corr = F.relu(vol_flat - baseline)
                    
                    # Deconvolution for all voxels
                    residue_funcs = torch.matmul(inv_aif_matrix, vol_corr)  # [T, H*W]
                    residue_funcs = F.relu(residue_funcs)  # Ensure non-negativity
                    
                    # CBF = max of residue function
                    cbf_flat = torch.max(residue_funcs, dim=0).values
                    
                    # CBV = area under curve
                    dt = self.time_values[1] - self.time_values[0] if self.time_values.shape[0] > 1 else 1.0
                    cbv_flat = torch.sum(vol_corr, dim=0) * dt
                    
                    # Reshape back
                    cbf_map[b_idx, 0] = cbf_flat.reshape(h, w)
                    cbv_map[b_idx, 0] = cbv_flat.reshape(h, w)
                    
                    # MTT = CBV/CBF
                    valid_mask = cbf_flat > 0
                    mtt_flat = torch.zeros_like(cbf_flat)
                    mtt_flat[valid_mask] = cbv_flat[valid_mask] / cbf_flat[valid_mask]
                    mtt_map[b_idx, 0] = mtt_flat.reshape(h, w)
                    
                    # Delay = time to peak
                    peak_idx = torch.argmax(vol_flat, dim=0)
                    delay_map[b_idx, 0] = (self.time_values[peak_idx]).reshape(h, w)
                    
                    # Tmax = time to max of residue function
                    tmax_idx = torch.argmax(residue_funcs, dim=0)
                    tmax_map[b_idx, 0] = (self.time_values[tmax_idx]).reshape(h, w)
                    
            except RuntimeError:
                # SVD can sometimes fail - use simplified calculations in that case
                print("SVD failed, using simplified perfusion calculations")
                # Basic calculations without deconvolution 
                if is_3d:
                    vol_slice = volume[b_idx, :, 0]  # [T, D, H, W]
                    # Peak enhancement as proxy for CBF
                    cbf_map[b_idx, 0] = torch.max(vol_slice, dim=0).values
                    # Area under curve for CBV
                    cbv_map[b_idx, 0] = torch.sum(vol_slice, dim=0) * dt
                    # Time to peak for Tmax
                    tmax_indices = torch.argmax(vol_slice, dim=0)
                    tmax_map[b_idx, 0] = self.time_values[tmax_indices]
                else:
                    vol_slice = volume[b_idx, :, 0] if volume.shape[2] > 0 else volume[b_idx]
                    cbf_map[b_idx, 0] = torch.max(vol_slice, dim=0).values
                    cbv_map[b_idx, 0] = torch.sum(vol_slice, dim=0) * dt
                    tmax_indices = torch.argmax(vol_slice, dim=0)
                    tmax_map[b_idx, 0] = self.time_values[tmax_indices]
        
        # Normalize maps to typical value ranges
        cbf_max = torch.quantile(cbf_map[cbf_map > 0], 0.99) if torch.sum(cbf_map > 0) > 0 else 1.0
        cbf_map = torch.clamp(cbf_map / cbf_max * 100, 0, 150)  # Scale to 0-150 ml/100g/min
        
        cbv_max = torch.quantile(cbv_map[cbv_map > 0], 0.99) if torch.sum(cbv_map > 0) > 0 else 1.0
        cbv_map = torch.clamp(cbv_map / cbv_max * 8, 0, 15)  # Scale to 0-15 ml/100g
        
        mtt_map = torch.clamp(mtt_map, 0, 20)  # Limit to 0-20 seconds
        tmax_map = torch.clamp(tmax_map, 0, 20)  # Limit to 0-20 seconds
        
        return {
            'cbf': cbf_map,
            'cbv': cbv_map,
            'mtt': mtt_map,
            'delay': delay_map,
            'tmax': tmax_map
        }
    
    def _clinical_threshold_loss(self, pred_maps, gt_maps):
        """
        Compute loss that emphasizes clinically relevant thresholds
        """
        loss = 0.0
        
        # CBF loss with emphasis on low-flow regions (ischemic core)
        for threshold in self.cbf_thresh:
            pred_mask = pred_maps['cbf'] < threshold
            gt_mask = gt_maps['cbf'] < threshold
            
            # Dice loss for threshold masks
            dice_loss = self._soft_dice_loss(pred_mask.float(), gt_mask.float())
            loss += dice_loss * 0.5
            
            # Also penalize value differences in these regions
            values_loss = F.l1_loss(
                pred_maps['cbf'][gt_mask | pred_mask], 
                gt_maps['cbf'][gt_mask | pred_mask]
            ) if torch.sum(gt_mask | pred_mask) > 0 else torch.tensor(0.0, device=pred_maps['cbf'].device)
            
            loss += values_loss * 0.5
        
        # Tmax loss with emphasis on perfusion delay regions (penumbra)
        for threshold in self.tmax_thresh:
            pred_mask = pred_maps['tmax'] > threshold
            gt_mask = gt_maps['tmax'] > threshold
            
            dice_loss = self._soft_dice_loss(pred_mask.float(), gt_mask.float())
            loss += dice_loss * 0.5
            
            values_loss = F.l1_loss(
                pred_maps['tmax'][gt_mask | pred_mask], 
                gt_maps['tmax'][gt_mask | pred_mask]
            ) if torch.sum(gt_mask | pred_mask) > 0 else torch.tensor(0.0, device=pred_maps['tmax'].device)
            
            loss += values_loss * 0.5
        
        return loss
    
    def _soft_dice_loss(self, pred, target, smooth=1.0):
        """Soft Dice loss"""
        intersection = torch.sum(pred * target)
        pred_sum = torch.sum(pred * pred)  # Using pred*pred instead of pred to make it smoother
        target_sum = torch.sum(target * target)
        
        dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
        return 1.0 - dice
    
    def forward(self, pred_volume, gt_volume):
        """
        Compute the perfusion map based loss
        
        Parameters:
        ----------
        pred_volume : torch.Tensor
            Predicted 4D volume of shape [B, T, C, H, W] or [B, T, C, D, H, W]
        gt_volume : torch.Tensor
            Ground truth 4D volume, same shape as pred_volume
            
        Returns:
        -------
        torch.Tensor
            Scalar loss value
        dict
            Breakdown of loss components
        """
        # Ensure inputs have proper dimensions
        assert pred_volume.shape == gt_volume.shape, f"Shape mismatch: {pred_volume.shape} vs {gt_volume.shape}"
        
        # Compute perfusion maps
        pred_maps = self._compute_perfusion_maps(pred_volume)
        gt_maps = self._compute_perfusion_maps(gt_volume)
        
        # Initialize loss components
        loss_components = {}
        total_loss = 0.0
        
        # CBF comparison (critical for tissue viability)
        cbf_loss = F.l1_loss(pred_maps['cbf'], gt_maps['cbf'])
        loss_components['cbf_loss'] = cbf_loss
        total_loss += cbf_loss * self.cbf_weight
        
        # Tmax comparison (critical for tissue at risk)
        tmax_loss = F.l1_loss(pred_maps['tmax'], gt_maps['tmax'])
        loss_components['tmax_loss'] = tmax_loss
        total_loss += tmax_loss * self.tmax_weight
        
        # CBV comparison (blood volume)
        cbv_loss = F.l1_loss(pred_maps['cbv'], gt_maps['cbv'])
        loss_components['cbv_loss'] = cbv_loss
        total_loss += cbv_loss * self.cbv_weight
        
        # MTT comparison (transit time)
        mtt_loss = F.l1_loss(pred_maps['mtt'], gt_maps['mtt'])
        loss_components['mtt_loss'] = mtt_loss
        total_loss += mtt_loss * self.mtt_weight
        
        # Structural similarity (for preserving edges and patterns)
        if self.struct_weight > 0:
            # SSIM loss for structural similarity
            structure_loss = 0.0
            for map_name in ['cbf', 'tmax']:
                # Ensure maps are properly formatted for SSIM
                p_map = pred_maps[map_name]
                g_map = gt_maps[map_name]
                
                # Normalize to 0-1 for SSIM calculation
                p_norm = (p_map - p_map.min()) / (p_map.max() - p_map.min() + 1e-6)
                g_norm = (g_map - g_map.min()) / (g_map.max() - g_map.min() + 1e-6)
                
                # Compute SSIM loss
                structure_loss += ssim_loss(p_norm, g_norm, window_size=11)
            
            structure_loss /= 2.0  # Average across maps
            loss_components['structure_loss'] = structure_loss
            total_loss += structure_loss * self.struct_weight
        
        # Clinical threshold-based loss
        if self.clinical_thresholds:
            threshold_loss = self._clinical_threshold_loss(pred_maps, gt_maps)
            loss_components['threshold_loss'] = threshold_loss
            total_loss += threshold_loss
        
        return total_loss, loss_components