"""The retained pointwise spatial DMPC architecture."""
import copy
import torch
from torch import nn


class LogPower(nn.Module):
    """Instantaneous spatial power; no pooling or temporal convolution."""
    def forward(self, values):
        # Square in float32 to avoid overflow with autocast half precision.
        return torch.log1p(values.float().square()).to(values.dtype)


class SpatialPointwiseEncoder(nn.Module):
    """Each time sample has an independent nonlinear spatial representation."""
    def __init__(self, channels, n_times, options, *, subject_count=0, subject_spatial=False):
        super().__init__()
        self.channels,self.n_times=channels,n_times
        self.options=copy.deepcopy(options)
        self.spatial_max_norm=float(options.get('spatial_max_norm',1.))
        self.classifier_max_norm=float(options.get('classifier_max_norm',.25))
        filters=options["spatial_filters"]
        self.latent_dim=filters*n_times
        self.spatial=nn.Conv1d(channels,filters,1,bias=False)
        self.norm=nn.BatchNorm1d(filters,eps=1e-3,momentum=.01)
        activation=options.get('activation','elu')
        if activation not in ('elu','log_power'):
            raise ValueError('Spatial activation must be elu or log_power')
        self.activation=nn.ELU() if activation=='elu' else LogPower()
        self.dropout=nn.Dropout(options["dropout"])
        self.subject_specific_spatial=subject_spatial
        self.subject_count=subject_count if subject_spatial else 0
        if subject_spatial:
            if type(subject_count) is not int or subject_count < 1:
                raise ValueError('Subject spatial adaptation needs the number of training participants')
            self.subject_residual=nn.Embedding(subject_count,filters*channels)
            nn.init.zeros_(self.subject_residual.weight)

    def project(self,eeg,subject_id=None):
        result=self.spatial(eeg)
        if self.subject_specific_spatial:
            if subject_id is None or subject_id.shape != (len(eeg),):
                raise ValueError('Subject spatial adaptation requires one participant ID per trial')
            residual=self.subject_residual(subject_id).reshape(len(eeg),self.spatial.out_channels,self.channels)
            result=result+torch.einsum('nfc,nct->nft',residual,eeg)
        return result

    def forward(self,eeg,subject_id=None):
        if eeg.ndim!=3 or eeg.shape[1:]!=(self.channels,self.n_times):
            raise ValueError("Spatial-only input axes differ from the registered EEG")
        return self.dropout(self.activation(self.norm(self.project(eeg,subject_id)))).flatten(1)

    def resolved_config(self):
        return {**self.options,"in_channels":self.channels,"n_times":self.n_times,
            "features":self.latent_dim,"pooled_time_bins":self.n_times,"temporal_receptive_field_samples":1,
            "learned_encoder_parameters":sum(p.numel() for p in self.parameters()),
            "subject_specific_spatial":self.subject_specific_spatial,"subject_count":self.subject_count,
            "batch_norm_eps":1e-3,"batch_norm_momentum":.01,
            "spatial_max_norm":self.spatial_max_norm,"classifier_max_norm":self.classifier_max_norm}



class SpatialTrialDecoder(nn.Module):
    """Spatial features at each time, followed by a whole-trial linear readout."""
    method="dmpc"

    def __init__(self,channels,n_times,classes,options,*,subject_count=0,subject_spatial=False):
        super().__init__()
        self.encoder=SpatialPointwiseEncoder(channels,n_times,options,subject_count=subject_count,
                                             subject_spatial=subject_spatial)
        self.head=nn.Linear(self.encoder.latent_dim,classes)
        nn.init.xavier_uniform_(self.encoder.spatial.weight)
        nn.init.xavier_uniform_(self.head.weight);nn.init.zeros_(self.head.bias)
        self.constrain_weights()

    def forward(self,eeg,subject_id):
        return self.head(self.encoder(eeg,subject_id))

    @torch.no_grad()
    def constrain_weights(self,*,head_only=False):
        constrained=[(self.head.weight,self.encoder.classifier_max_norm)]
        if not head_only:constrained.insert(0,(self.encoder.spatial.weight,self.encoder.spatial_max_norm))
        for weight,limit in constrained:
            norm=weight.flatten(1).norm(dim=1).clamp_min(1e-12)
            weight.mul_((limit/norm).clamp_max(1).view(-1,*([1]*(weight.ndim-1))))
        encoder=self.encoder
        if encoder.subject_specific_spatial and not head_only:
            shared=encoder.spatial.weight[:,:,0]
            residual=encoder.subject_residual.weight.view(encoder.subject_count,*shared.shape)
            effective=shared[None]+residual
            norm=effective.norm(dim=2,keepdim=True).clamp_min(1e-12)
            effective.mul_((encoder.spatial_max_norm/norm).clamp_max(1))
            residual.copy_(effective-shared[None])


def build_dmpc_model(base,config):
    if config['model']['encoder']['type']!='spatial_only':
        from .alternative_models import AlternativeTrialDecoder
        if config['model']['subject_conditioning'].get('enabled',False):
            raise ValueError('Alternative DMPC encoders use shared participant-independent filters')
        return AlternativeTrialDecoder(base.input_channels,len(base.time_s),base.class_count,config['model']['encoder'])
    subject_count=max(base.participant_indices)+1
    return SpatialTrialDecoder(base.input_channels,len(base.time_s),base.class_count,config['model']['encoder'],
        subject_count=subject_count,subject_spatial=config['model']['subject_conditioning'].get('enabled',False))

