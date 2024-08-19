from .cfd_datamodule import CFDSDFDataModule, CFDDataModule, AhmedBodyDataModule


def instantiate_datamodule(config):
    if config.data_module == "CFDDataModule":
        return CFDDataModule(
            config.data_path,
        )
    elif config.data_module == "CFDSDFDataModule":
        assert config.sdf_spatial_resolution is not None
        return CFDSDFDataModule(
            config.data_path,
            spatial_resolution=config.sdf_spatial_resolution,
        )
    elif config.data_module == "AhmedBodyDataModule":
        assert config.sdf_spatial_resolution is not None
        return AhmedBodyDataModule(
            config.data_path,
            spatial_resolution=config.sdf_spatial_resolution,
        )
    else:
        raise NotImplementedError(f"Unknown datamodule: {config.data_module}")
