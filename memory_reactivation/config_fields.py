"""Accepted Target request fields, including explicit legacy aliases and snapshots.

Leaf value/range validation belongs to the owning model/data/training component.
This boundary rejects unknown names before defaults can silently hide a typo.
"""
from collections.abc import Mapping
from difflib import get_close_matches


def _fields(names):
    return frozenset(names.split())


FIELDS = {
    "": _fields("""schema_version seed task lag data model train plots run swanlab preprocessing
        cv_fold cv_folds cv_strategy cv_split_dir cv_split_sha256 cv_val_blocks cv_train_subjects
        cv_val_subjects run_mode output_root output_dir"""),
    "data": _fields("""manifest_csv split_csv normalization_json normalization_mode experiments subjects lags
        require_clean_history balance_train_cells balance_eval_cells balance_seed batch_sampling cache_size
        offline_cache_dir num_workers time_window_start_s time_window_end_s choice_cell_history_fields
        train_cell_weighting train_cell_weight_max train_drop_incomplete_cells train_dynamic_cell_sampling
        train_min_cell_class_count train_majority_ratio_cap train_label_shuffle_within_cells
        train_label_shuffle_method train_label_shuffle_seed train_time_shuffle_within_train train_time_shuffle_seed
        eval_min_cell_class_count pitch_class_indices dynamic_time_window_samples dynamic_time_window_sample_options
        dynamic_time_window_random dynamic_time_window_seed dynamic_time_window_deterministic
        graph_laplacian_json graph_laplacian_sha256 graph_laplacian_mode cv_fold cv_folds cv_split_dir cv_strategy
        cv_split_sha256 cv_val_blocks cv_train_subjects cv_val_subjects"""),
    "model": _fields("""encoder resolved_encoder attention_dim attention_dropout attention_mode attention_query_source
        attention_score_centering attention_gate_centering attention_gate_activation attention_gate_combination
        attention_effective_normalization attention_gate_init_bias pool_valid_context_only pooled_head_mode
        classifier_num_layers classifier_hidden_dim classifier_dropout subject_conditioning fusion reconstruction
        time_source_block time_head_mode time_no_class_enabled resolved_time_point_count resolved_time_axis_sha256
        resolved_task_class_indices pretrained_dpc freeze"""),
    "model.encoder": _fields("""type encoder_type in_channels latent_dim num_blocks num_layers kernel_size dropout
        dilations expansion_ratio ffn_ratio window_size time_point_count num_heads shift_size attention_pattern
        window_partition position_encoding_mode boundary_mode boundary_noise_seed hidden_dim receptive_field_samples
        left_context_samples right_context_samples"""),
    "model.subject_conditioning": _fields("""enabled mode embedding_dim condition_dropout adaln_dropout dropout
        resolved_training_subject_ids resolved_training_subject_names"""),
    "model.fusion": _fields("enabled hidden_dim dropout"),
    "model.reconstruction": _fields("enabled"),
    "model.resolved_task_class_indices": _fields("choice pitch motor"),
    "model.pretrained_dpc": _fields("""enabled checkpoint load_subject_conditioning resolved_checkpoint
        checkpoint_sha256 loaded_components loaded_key_count encoder_config_checked"""),
    "model.freeze": _fields("encoder subject_conditioning trainable_parameter_count total_parameter_count"),
    "train": _fields("""iterations epochs batch_size optimizer learning_rate weight_decay momentum dampening nesterov
        muon lr_schedule lr_schedule_iterations min_learning_rate min_lr warmup_iterations warmup_epochs
        wd_schedule wd_warmup_iterations min_weight_decay validation_interval_iterations log_interval_iterations
        patience_iterations checkpoint_metric checkpoint_mode checkpoint_start_iteration early_stopping_start_iteration
        trajectory_checkpoint_iterations amp grad_clip_norm progress_bar label_smoothing ordinal_sigma
        loss_type bce_pos_weight focal_gamma focal_alpha
        time_loss_type time_bce_pos_weight time_focal_gamma time_focal_alpha time_loss_weight
        pooled_loss_type pooled_bce_pos_weight pooled_focal_gamma pooled_focal_alpha pooled_loss_weight
        fusion_loss_type fusion_bce_pos_weight fusion_focal_gamma fusion_focal_alpha fusion_loss_weight
        train_time_start_s train_time_end_s attention_time_start_s attention_time_end_s
        attention_topk_k attention_topk_ratio all_time_head_warmup_iterations
        invalid_time_loss_weight invalid_time_loss_scope topk_covered_time_loss_weight
        attention_mask_penalty_weight attention_mask_penalty_mode attention_mass_penalty_weight
        clean_attention_penalty_weight clean_attention_penalty_probability attention_smoothness_weight
        attention_smoothness_mode non_topq_time_loss_probability matched_null_attention_weight matched_null_noise_mode
        matched_null_min_noise_std att_time_consistency_weight real_noise_margin real_noise_margin_weight
        reconstruction_loss_weight near_full_noise temporal_crop_noise augmentation_pipeline"""),
    "train.muon": _fields("learning_rate momentum ns_steps nesterov"),
    "train.near_full_noise": _fields("enabled probability coverage_ratio time_loss_scale"),
    "train.augmentation_pipeline": _fields("enabled roll_probability roll_max_shift_s roll_left_probability"),
    "train.temporal_crop_noise": _fields("""enabled geometry mode probability side min_duration_s max_left_duration_s
        max_right_duration_s block_duration_s min_block_duration_s max_block_duration_s max_duration_s
        min_blocks max_blocks min_total_duration_s max_total_duration_s min_clean_gap_s allow_edge_blocks
        noise_mode noise_mode_probabilities min_noise_std sample_interval_s sample_rate_hz supervision_mode"""),
    "train.temporal_crop_noise.noise_mode_probabilities": _fields(
        "sample_channel batch_channel standard_normal time_shuffle time_shuffle_independent"),
    "plots": _fields("enabled splits output_dir activation gaussian_baseline event_markers time_axis"),
    "plots.activation": _fields("enabled threshold"),
    "plots.gaussian_baseline": _fields("enabled seed noise_mode"),
    "plots.time_axis": _fields("start_s end_s"),
    "plots.event_markers": _fields("enabled markers"),
    "plots.event_markers.markers[]": _fields("time time_s label color alpha"),
    "run": _fields("base_output_dir base_experiment_name experiment_name name_suffix"),
    "swanlab": _fields("""enabled mode project workspace experiment_name base_experiment_name
        base_tags tags description group"""),
    "preprocessing": _fields("graph_laplacian"),
    "preprocessing.graph_laplacian": _fields("enabled mode n_neighbors electrodes_npz sigma_m"),
}
FIELDS["model.resolved_encoder"] = FIELDS["model.encoder"]
# These are identifier -> value mappings, not option dictionaries.
IDENTITY_MAPPINGS = frozenset({"cv_val_blocks", "data.cv_val_blocks",
                              "model.subject_conditioning.resolved_training_subject_names"})


def validate_request_fields(value, path=""):
    if path in IDENTITY_MAPPINGS:
        if not isinstance(value, Mapping):
            raise ValueError(f"{path} must be a mapping")
        return
    fields = FIELDS.get(path)
    if fields is None:
        if isinstance(value, Mapping):
            raise ValueError(f"{path} does not accept nested configuration fields")
        return
    if value is None and path not in {"", "data", "model", "train"}:
        return  # Preserve existing optional-section null semantics.
    if not isinstance(value, Mapping):
        raise ValueError(f"{path or 'Configuration'} must be a mapping")
    for key, item in value.items():
        child = f"{path}.{key}" if path else str(key)
        if key not in fields:
            suggestions = get_close_matches(str(key), sorted(fields), n=1)
            hint = f"; did you mean {suggestions[0]!r}?" if suggestions else ""
            raise ValueError(f"Unknown request configuration field: {child}{hint}")
        if child == "plots.event_markers.markers":
            if not isinstance(item, (list, tuple)):
                raise ValueError(f"{child} must be a list")
            for marker in item:
                validate_request_fields(marker, child + "[]")
        else:
            validate_request_fields(item, child)
