"""Control parameters and incomplete exports must never silently relabel old results."""

import importlib
import json
from pathlib import Path
import shutil
import sys

import pytest
import torch
import yaml

from pb2023_eeg.artifacts import exports as export_integrity
from pb2023_eeg.training.export_integrity import describe_export
from pb2023_eeg.training.target import export_target_outputs
from pb2023_eeg.workflows.reexport import copy_existing_target_exports
from pb2023_eeg.workflows.train import build_model as build_target


def contents(directory):
    return {str(p.relative_to(directory)): p.read_bytes() for p in directory.rglob('*') if p.is_file()}


def small_model_config(channels=2, fusion=True):
    return dict(encoder=dict(type='cnn', in_channels=channels, latent_dim=4, num_blocks=1,
                            kernel_size=3, expansion_ratio=1, dropout=0.),
                classifier_num_layers=1, classifier_dropout=0., attention_mode='scorer',
                attention_dim=4, attention_dropout=0., fusion=dict(enabled=fusion))


@pytest.fixture(params=['target'])
def exports(request, tmp_path):
    decoder = request.param
    build, export, copy = ((build_target, export_target_outputs, copy_existing_target_exports)
                          if decoder == 'target' else
                          (build_time, export_time_outputs, copy_existing_time_exports))
    torch.manual_seed(19)
    batch = dict(eeg=torch.randn(12, 2, 8), target=torch.arange(12) % 2,
                 subject_id=torch.zeros(12, dtype=torch.long), record_id=torch.zeros(12, dtype=torch.long),
                 trial0=torch.arange(12), exp=torch.ones(12, dtype=torch.long),
                 lag=torch.ones(12, dtype=torch.long), current_stimulus=torch.arange(12) // 4 + 1)

    def create(directory='source', input_kind='real', fusion=True, trial_scores=False, **overrides):
        model = build(small_model_config(fusion=fusion), time_point_count=8).eval()
        arguments = dict(model=model, loader=[batch], task='choice', device=torch.device('cpu'),
                         amp=False, output_dir=tmp_path/directory, split='val',
                         time_s=torch.arange(8, dtype=torch.float32)/100,
                         progress_bar_enabled=False, input_kind=input_kind,
                         time_score_mode='softmax', gaussian_seed=0,
                         time_shuffle_seed=101, time_roll_shift_samples=3)
        if decoder == 'target':
            arguments['attention_indices'] = torch.arange(8)
        else:
            arguments.update(fusion_indices=torch.arange(8), export_trial_time_scores=trial_scores)
        arguments.update(overrides)
        export(**arguments)
        return arguments

    return decoder, export, copy, create


@pytest.mark.parametrize('input_kind', ['real', 'gaussian', 'time_shuffle', 'time_roll'])
def test_matching_export_is_reused_byte_for_byte(exports, tmp_path, input_kind):
    _, export, copy, create = exports
    arguments = create(input_kind=input_kind)
    expected = describe_export(export, **arguments)
    source, destination = arguments['output_dir'], tmp_path/'destination'
    copied = copy(source, destination, splits=['val'], input_kinds=[input_kind],
                  expected_parameters={('val', input_kind): expected})
    assert copied and contents(source) == contents(destination)


@pytest.mark.parametrize('input_kind,changed', [
    ('time_roll', dict(time_roll_shift_samples=7)),
    ('time_shuffle', dict(time_shuffle_seed=201)),
    ('gaussian', dict(gaussian_seed=1)),
    ('gaussian', dict(gaussian_noise_mode='global')),
    ('real', dict(activation_threshold=.7)),
    ('real', dict(min_cell_class_count=2)),
    ('real', dict(time_score_mode='sigmoid')),
    ('real', dict(time_s=torch.arange(8, dtype=torch.float32)/50)),
])
def test_mismatched_parameters_fail_before_destination_changes(exports, tmp_path, input_kind, changed):
    _, export, copy, create = exports
    arguments = create(input_kind=input_kind)
    destination = tmp_path/'destination'
    shutil.copytree(arguments['output_dir'], destination)
    before = contents(destination)
    expected = describe_export(export, **{**arguments, **changed})
    with pytest.raises(ValueError, match='parameter mismatch'):
        copy(arguments['output_dir'], destination, splits=['val'], input_kinds=[input_kind],
             expected_parameters={('val', input_kind): expected})
    assert contents(destination) == before


def test_changed_checkpoint_weights_are_not_reused(exports, tmp_path):
    _, export, copy, create = exports
    arguments = create()
    with torch.no_grad():
        next(arguments['model'].parameters()).add_(1)
    with pytest.raises(ValueError, match='model_state_sha256'):
        copy(arguments['output_dir'], tmp_path/'destination', splits=['val'], input_kinds=['real'],
             expected_parameters={('val', 'real'): describe_export(export, **arguments)})
    assert not (tmp_path/'destination').exists()


def test_csv_only_legacy_exports_require_regeneration(exports, tmp_path):
    _, _, copy, create = exports
    arguments = create(input_kind='time_roll')
    (arguments['output_dir']/'val_time_roll_export_manifest.json').unlink()
    with pytest.raises(ValueError, match='Re-export without --reuse-existing-kinds'):
        copy(arguments['output_dir'], tmp_path/'destination', splits=['val'], input_kinds=['time_roll'])
    assert not (tmp_path/'destination').exists()


@pytest.mark.parametrize('damage', ['missing_summary', 'modified_csv', 'missing_inventory_entry'])
def test_incomplete_or_changed_csvs_preserve_existing_destination(exports, tmp_path, damage):
    _, _, copy, create = exports
    arguments = create()
    source, destination = arguments['output_dir'], tmp_path/'destination'
    shutil.copytree(source, destination)
    before = contents(destination)
    if damage == 'missing_summary':
        (source/'val_time_cell_summary.csv').unlink()
    elif damage == 'modified_csv':
        path = source/'val_time_accuracy.csv'
        # Changing content need not change the file size.
        path.write_bytes(path.read_bytes().replace(b'choice', b'motor'))
    else:
        path = source/'val_export_manifest.json'
        manifest = json.loads(path.read_text())
        del manifest['files']['val_time_cell_summary.csv']
        path.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, FileNotFoundError)):
        copy(source, destination, splits=['val'], input_kinds=['real'])
    assert contents(destination) == before


def test_later_incomplete_condition_does_not_publish_earlier_condition(exports, tmp_path):
    _, _, copy, create = exports
    first = create(input_kind='real')
    create(input_kind='time_roll')
    destination = tmp_path/'destination'
    destination.mkdir()
    (destination/'val_time_accuracy.csv').write_text('previous output')
    (first['output_dir']/'val_time_roll_time_cell_summary.csv').unlink()
    before = contents(destination)
    with pytest.raises(FileNotFoundError, match='time_cell_summary'):
        copy(first['output_dir'], destination, splits=['val'], input_kinds=['real', 'time_roll'])
    assert contents(destination) == before


@pytest.mark.parametrize('failure', ['copy', 'publish'])
def test_io_failure_keeps_previous_export_intact(exports, tmp_path, monkeypatch, failure):
    _, _, copy, create = exports
    arguments = create()
    source, destination = arguments['output_dir'], tmp_path/'destination'
    shutil.copytree(source, destination)
    before = contents(destination)
    if failure == 'copy':
        original = export_integrity.shutil.copy2
        def fail_late(source, destination, *args, **kwargs):
            if Path(source).name.endswith('time_cell_summary.csv'):
                raise OSError('injected copy failure')
            return original(source, destination, *args, **kwargs)
        monkeypatch.setattr(export_integrity.shutil, 'copy2', fail_late)
    else:
        original = Path.replace
        def fail_late(path, target):
            if '.export-reuse-' in str(path) and path.name.endswith('time_cell_summary.csv'):
                raise OSError('injected publish failure')
            return original(path, target)
        monkeypatch.setattr(Path, 'replace', fail_late)
    with pytest.raises(OSError, match='injected'):
        copy(source, destination, splits=['val'], input_kinds=['real'])
    assert contents(destination) == before


def test_failed_fresh_export_keeps_previous_csvs_and_provenance(exports, monkeypatch):
    decoder, export, _, create = exports
    arguments = create()
    before = contents(arguments['output_dir'])
    module = importlib.import_module('pb2023_eeg.evaluation.export')
    original = module.write_rows_csv
    def fail_late(path, rows):
        if path.name.endswith('time_cell_summary.csv'):
            raise OSError('injected export failure')
        return original(path, rows)
    monkeypatch.setattr(module, 'write_rows_csv', fail_late)
    with pytest.raises(OSError, match='injected'):
        export(**arguments)
    assert contents(arguments['output_dir']) == before




@pytest.mark.parametrize('decoder', ['target'])
def test_real_training_and_cli_reuse_validate_parameters_and_preserve_failed_output(tmp_path, monkeypatch, decoder):
    # Use the existing synthetic EEG fixture and actual training/checkpoint/export paths.
    spec = importlib.util.spec_from_file_location('reuse_smoke_fixture', Path(__file__).with_name('helpers_smoke.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fixture = module.TargetPipelineSmokeTest()
    fixture.setUp()
    try:
        config = dict(seed=42,
            data=dict(manifest_csv=str(fixture.manifest), split_csv=str(fixture.splits),
                      normalization_json=str(fixture.normalization), experiments=[1], lags=[1],
                      balance_train_cells=False, balance_eval_cells=False, num_workers=0,
                      offline_cache_dir=str(tmp_path/'cache')),
            model=small_model_config(channels=64),
            train=dict(iterations=2, validation_interval_iterations=1, log_interval_iterations=1,
                       batch_size=16, learning_rate=.001, lr_schedule='constant', amp=False,
                       progress_bar=False, loss_type='ce', time_loss_weight=.5,
                       pooled_loss_weight=1., fusion_loss_weight=.2,
                       train_time_start_s=-.1, train_time_end_s=.29,
                       attention_time_start_s=-.1, attention_time_end_s=.29),
            plots=dict(enabled=False, splits=['val'], gaussian_baseline=dict(enabled=False)),
            swanlab=dict(enabled=False),
            run=dict(base_output_dir=str(tmp_path/'runs'), base_experiment_name='reuse-test'))
        filename = tmp_path/'config.yaml'
        filename.write_text(yaml.safe_dump(config))
        monkeypatch.setenv('PB2023_SKIP_CODE_SNAPSHOT', '1')
        monkeypatch.setenv('PB2023_SWANLAB_ENABLED', '0')
        training = importlib.import_module('pb2023_eeg.workflows.train')
        monkeypatch.setattr(sys, 'argv', ['train', '--config', str(filename), '--task', 'choice', '--lag', '1'])
        training.main()
        run = next((tmp_path/'runs').rglob('best_val.ckpt')).parent
        reexport = importlib.import_module('pb2023_eeg.workflows.reexport')
        # Disable only rendering; inference and all file checks run normally.
        monkeypatch.setattr(reexport, 'plot_target_conditions', lambda *args, **kwargs: None)
        # Reuse the untouched training export before generating any reexports.
        direct = tmp_path/'direct-reuse'
        reexport.main(['--run-dir', str(run), '--splits', 'val', '--input-kinds', 'real',
                       '--reuse-existing-kinds', 'real', '--controls-output-dir', str(direct)])
        for path in (direct/'exports').iterdir():
            assert path.read_bytes() == (run/path.name).read_bytes()
        base = ['reexport', '--run-dir', str(run), '--splits', 'val', '--input-kinds',
                'real', 'time_roll', 'time_shuffle']
        same = ['--time-roll-shift-s', '.03', '--time-shuffle-seed', '100']
        original = tmp_path/'original'
        monkeypatch.setattr(sys, 'argv', base + same + ['--controls-output-dir', str(original)])
        reexport.main()
        for path in (original/'exports').iterdir():
            shutil.copy2(path, run/path.name)
        output = tmp_path/'reused'
        reuse = ['--controls-output-dir', str(output), '--reuse-existing-kinds',
                 'real', 'time_roll', 'time_shuffle']
        monkeypatch.setattr(sys, 'argv', base + same + reuse)
        reexport.main()
        assert contents(original/'exports') == contents(output/'exports')
        before = contents(output)
        for flags, error in [(['--time-roll-shift-s', '.07', '--time-shuffle-seed', '100'], 'time_roll_shift_samples'),
                             (['--time-roll-shift-s', '.03', '--time-shuffle-seed', '200'], 'time_shuffle_seed')]:
            monkeypatch.setattr(sys, 'argv', base + flags + reuse)
            with pytest.raises(ValueError, match=error):
                reexport.main()
            assert contents(output) == before
        (run/'val_time_roll_time_cell_summary.csv').unlink()
        monkeypatch.setattr(sys, 'argv', base + same + reuse)
        with pytest.raises(FileNotFoundError, match='time_cell_summary'):
            reexport.main()
        assert contents(output) == before
    finally:
        fixture.tearDown()
