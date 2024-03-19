gss_optimization_config = {
    "xyz": {
        "optimizer": AdamOptimizerConfig(lr=0.00016, eps=1e-15),
        "scheduler": ExponentialDecaySchedulerConfig(
            lr_final=0.0000016,
            max_steps=30000,
        ),
    },
    "features_dc": {
        "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
        "scheduler": None,
    },
    "features_rest": {
        "optimizer": AdamOptimizerConfig(lr=0.000125, eps=1e-15),
        "scheduler": None,
    },
    "opacity": {
        "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
        "scheduler": None,
    },
    "scaling": {
        "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
        "scheduler": None,
    },
    "rotation": {
        "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
        "scheduler": None,
    },
}

method_configs["gss"] = TrainerConfig(
    method_name="gss",
    steps_per_save=10000,
    steps_per_eval_batch=1000000,
    steps_per_eval_image=1000000,
    steps_per_eval_all_images=1000000,
    max_num_iterations=30000,
    mixed_precision=False,
    use_grad_scaler=False,
    optimizers=gss_optimization_config,
    pipeline=VanillaPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(
                auto_scale_poses=False,
                center_method="none",
                orientation_method="none",
            ),
            camera_optimizer=CameraOptimizerConfig(mode="off"),
            image_based_pipeline=True,
        ),
        model=GaussianSplatConfig(),
    ),
)
