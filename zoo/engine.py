import torch

from .logger import Logger

def warmup_cosine_decay(
    optimizer,
    max_epochs: int,
    warmup_epochs: int | float = 0,
    warmup_min: float = 0,
    warmup_max: float = 1,
    cosine_min: float = 0,
    cosine_max: float = 1,
    loader_size: int = 1,
):
    import math
    from torch.optim.lr_scheduler import LambdaLR
    warmup_steps = int(warmup_epochs * loader_size)
    warmup_amplitude = warmup_max - warmup_min
    cosine_steps = max_epochs * loader_size - warmup_steps
    cosine_amplitude = 0.5 * (cosine_max - cosine_min)
    cosine_mean = 0.5 * (cosine_max + cosine_min)
    def cosine_decay_with_warmup(i):
        if warmup_steps and i < warmup_steps:
            return warmup_amplitude * i / warmup_steps + warmup_min
        else:
            i = i - warmup_steps
            return cosine_amplitude * math.cos(i * math.pi / cosine_steps) + cosine_mean
    return LambdaLR(optimizer, cosine_decay_with_warmup)

_grad_scaler = None
_bs_accumulated = None

def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    grad_clip: float = None,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
    accumulate_batch_size: int = None,
    logger: Logger = None,
    device: str = "cuda",
    epoch: int = 0,
    max_epochs: int = 0,
    use_amp: bool = False,
):
    global _grad_scaler, _bs_accumulated

    model.train()
    if use_amp and _grad_scaler is None:
            _grad_scaler = torch.GradScaler(device=device)
    if accumulate_batch_size is not None:
        _bs_accumulated = 0
    
    for step, (coors, masks) in enumerate(loader, 1):

        coors = coors.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.autocast(device_type=device, enabled=use_amp):
            preds = model(coors)
            loss = criterion(preds, coors, mask=masks)

        if use_amp:
            _grad_scaler.scale(loss).backward()
        else:
            loss.backward()

        if accumulate_batch_size is not None:
            _bs_accumulated += coors.shape[0]

        if accumulate_batch_size is None or _bs_accumulated == accumulate_batch_size:
            if grad_clip is not None:
                if use_amp:
                    _grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            if use_amp:
                _grad_scaler.step(optimizer)
                _grad_scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

            if accumulate_batch_size is not None:
                _bs_accumulated = 0

        if device == "cuda":
            torch.cuda.synchronize()

        if logger is not None:
            loss = loss.item()
            lr = optimizer.param_groups[0]["lr"]
            logger.add(
                loss=(loss, dict(trace=True, fmt=".4f", tag="train")),
                lr=(lr, dict(trace=False, fmt=".3e"))
            )
            logger.commit(
                epoch=epoch, max_epochs=max_epochs,
                step=step, max_steps=len(loader),
            )

        if lr_scheduler is not None:
            lr_scheduler.step()

@torch.no_grad()
def val(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    logger: Logger = None,
    device: str = "cuda",
    epoch: int = 0,
    max_epochs: int = 0,
):
    model.eval()
    culoss, cucnt = 0, 0
    for step, (coors, masks) in enumerate(loader, 1):
        coors = coors.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        preds = model(coors)
        loss = criterion(preds, coors, mask=masks)

        if device == "cuda":
            torch.cuda.synchronize()
        
        culoss += loss.item() * coors.shape[0]
        cucnt += coors.shape[0]

    loss = culoss / cucnt
    if logger is not None:
        logger.add(
            loss=(loss, dict(trace=True, fmt=".4f", tag="val")),
        )
        logger.commit(
            epoch=epoch, max_epochs=max_epochs,
        )
    else:
        print(f"Validation loss: {loss:.4f}")
