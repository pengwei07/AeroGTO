from .GNOFNOGNO import GNOFNOGNOAhmed
def instantiate_network(config):
    #out_channels = 1  # pressure
    print(config.model)
    if config.model == "GNOFNOGNOAhmed":
        print("using GNOFNOGNOAhmed")
        model = GNOFNOGNOAhmed(
            radius_in=config.radius_in,
            radius_out=config.radius_out,
            embed_dim=config.embed_dim,
            hidden_channels=config.hidden_channels,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            fno_modes=config.fno_modes,
            fno_hidden_channels=config.fno_hidden_channels,
            fno_out_channels=config.fno_hidden_channels,
            fno_domain_padding=0.125,
            fno_norm="ada_in",
            fno_factorization="tucker",
            fno_rank=0.4,
            linear_kernel=True,
            weighted_kernel=config.weighted_kernel,
            subsample_train=config.subsample_train,
            subsample_eval=config.subsample_eval,
            max_in_points=config.max_in_points,
        )
    print("The model size is ", count_params(model))
    return model

