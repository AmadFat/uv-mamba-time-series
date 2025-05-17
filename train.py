import zoo
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

import zoo.criterion
import time

def main():
    max_epochs = 10

    dataset = zoo.data.MPTSDataset(
        root="./data",
        train=True,
        transform=zoo.data.Normalize(),
    )
    train_dataset, val_dataset = dataset.split([0.8, 0.2])
    train_loader = zoo.data.MPTSLoader(
        dataset=train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=24,
        pin_memory=False,
        drop_last=True,
        collate_fn=zoo.data.collate_fn,
    )
    val_loader = zoo.data.MPTSLoader(
        dataset=val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=8,
        pin_memory=False,
        drop_last=False,
        collate_fn=zoo.data.collate_fn,
    )
    model = zoo.model.MPTSModel(
        d_model=64,
        num_blocks=2,
        block_norm=partial(nn.RMSNorm, eps=1e-6),
        mamba_d_state=64,
        mamba_d_conv=2,
        mamba_expand=2,
        mamba_headdim=16,
        mlp_expand=4,
        mlp_bias=True,
        mlp_activation=nn.SiLU,
        mlp_norm=None,
        mlp_dropout=0,
        mlp_depth=2,
        device="cuda",
        dtype=None,
    ).cuda()
    model.compile()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-3,
        betas=(0.95, 0.999),
    )
    scheduler = zoo.engine.warmup_cosine_decay(
        optimizer,
        max_epochs=max_epochs,
        warmup_epochs=2,
        loader_size=len(train_loader),
    )
    criterion = zoo.criterion.MaskedMSELoss()
    logger = zoo.logger.Logger(
        name="MPTS",
        step_interval=5,
        file_path=f"./logs/{time.strftime('%Y-%m-%d_%H-%M-%S')}.log",
        tb_path=f"./tbevents/{time.strftime('%Y-%m-%d_%H-%M-%S')}",
    )

    for epoch in range(max_epochs):
        zoo.engine.train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            logger=logger,
            device="cuda",
            epoch=epoch,
            max_epochs=max_epochs,
            use_amp=True,
        )

        zoo.engine.val(
            model=model,
            loader=val_loader,
            criterion=criterion,
            logger=logger,
            device="cuda",
            epoch=epoch,
            max_epochs=max_epochs,
        )

if __name__ == "__main__":
    main()
