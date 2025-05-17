import zoo
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

import zoo.criterion
import time

def main():
    max_epochs = 6
    exp = time.strftime("%Y-%m-%d_%H-%M-%S")

    dataset = zoo.data.MPTSDataset(
        root="./data",
        train=True,
        transform=zoo.data.Normalize(),
    )
    train_dataset, val_dataset = dataset.split([0.5, 0.5])
    train_loader = zoo.data.MPTSLoader(
        dataset=train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=24,
        pin_memory=False,
        drop_last=True,
        collate_fn=zoo.data.collate_fn,
    )
    val_loader = zoo.data.MPTSLoader(
        dataset=val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=8,
        pin_memory=False,
        drop_last=False,
        collate_fn=zoo.data.collate_fn,
    )
    model = zoo.model.MPTSModel(
        d_model=16,
        num_blocks=2,
        block_norm=partial(nn.RMSNorm, eps=1e-6),
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_expand=2,
        mamba_headdim=4,
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
        warmup_epochs=1,
        loader_size=len(train_loader),
    )
    criterion = zoo.criterion.MaskedMSELoss()
    logger = zoo.logger.Logger(
        name="MPTS",
        step_interval=5,
        file_path=f"./logs/{exp}.log",
        tb_path=f"./tbevents/{exp}",
        mode="DEBUG",
    )
    logger.add(num_param=(sum(p.numel() for p in model.parameters()), dict(trace=False, fmt="d")))
    logger.commit()

    for epoch in range(1, 1 + max_epochs):
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
    
    from pathlib import Path
    Path("./ckpts").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"./ckpts/{exp}.pth")

if __name__ == "__main__":
    main()
