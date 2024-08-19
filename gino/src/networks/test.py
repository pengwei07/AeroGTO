from GNOFNOGNO import GNOFNOGNOAhmed


model = GNOFNOGNOAhmed(
            radius_in=0.035,
            radius_out=0.035,
            # radius_in=0.055,
            # radius_out=0.055,
            embed_dim=30,
            #hidden_channels=(64, 64),
            hidden_channels=(20, 20),
            in_channels=2,
            out_channels=1,
            # fno_modes=(32, 32, 32),
            fno_modes=(12, 12, 12),
            # fno_hidden_channels=64,
            fno_hidden_channels=20,
            #fno_out_channels=64,
            fno_out_channels=20,
            # fno_domain_padding=0.125,
            fno_norm="ada_in",
            fno_factorization="tucker",
            fno_rank=0.4,
            linear_kernel=True,
            weighted_kernel=False,
            subsample_train=1,
            subsample_eval=1,
        )

