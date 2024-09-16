_base_ = ['../_base_/default_runtime.py']

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=300),
    checkpoint=dict(type='CheckpointHook', interval=10))
custom_hooks = [dict(type='LogModelHook')]

vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(type='Visualizer', vis_backends=vis_backends)

train_dataloader = dict(batch_size=2,
                        dataset=dict(type='SubwayDataset',
                                     data_path='./data/subway/train_data.pkl'),
                        sampler=dict(type='DefaultSampler'),
                        collate_fn=dict(type='default_collate'),
                        num_workers=2)
val_dataloader = dict(batch_size=16,
                      dataset=dict(type='SubwayDataset',
                                     data_path='./data/subway/val_data.pkl'),
                      sampler=dict(type='DefaultSampler', shuffle=False),
                      collate_fn=dict(type='default_collate'),
                      num_workers=2)
test_dataloader = dict(batch_size=16,
                       dataset=dict(type='SubwayDataset',
                                    data_path='./data/subway/all_data.pkl'),
                       sampler=dict(type='DefaultSampler', shuffle=False),
                       collate_fn=dict(type='default_collate'),
                       num_workers=2)

model = dict(type='SubwayMLP',
             sample_points=640,
             num_features=16,
             num_classes=8,
             hidden_feat=64,
             hidden_layers=2)

max_epoch = 500
warmup_epoch = max_epoch // 10
randomness = dict(seed=2022)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epoch, val_interval=10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
val_evaluator = dict(type='MyEvaluator', 
                     metrics=[dict(type='MyAcc'), 
                              dict(type='MyDumpResults', 
                                   out_file_path='./work_dirs/result.pkl')])
test_evaluator = val_evaluator

optim_wrapper = dict(type='OptimWrapper', optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.01))
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.01,
        by_epoch=True,
        begin=0,
        end=warmup_epoch,),
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=warmup_epoch,
        end=max_epoch,
    )
]
