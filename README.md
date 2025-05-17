# Mamba-Pinn-Typhoon

## Prerequisites

pixi: https://pixi.sh/latest/advanced/installation/

```shell
# If you add some packages or change the dependency tree,
# export the lock file first, this ensures the sources can
# be used for cross-platform usage or redistribution.
# I recommend use `-vvv` as "very verbose", because in this
# step pixi stucks sometimes, may it's something will be
# better in the future.
pixi lock -vvv 

# For environment syncronizing or dependencies installation.
# If you did not change the lock file as I provided, the
# downloading and building will be `fast` here.
pixi install

# You can just view `pixi run` as a powerful environment
# exportation. E.g., `python train.py` -> 'pixi run python
# train.py'. With no explicit symbol link `python` in my
# environment, I got this:
# af@laptop-omen9:~/Projects/mamba-pinn-typhoon$ python train.py 
# Command 'python' not found, did you mean:
#   command 'python3' from deb python3
#   command 'python' from deb python-is-python3
pixi run python train.py
```

## Project Structure

```
├── assets
├── data
├── logs
├── pixi.lock
├── pixi.toml
├── README.md
├── tbevents
├── test.py
├── train.py
└── zoo
    ├── criterion.py
    ├── data.py
    ├── engine.py
    ├── __init__.py
    ├── logger.py
    └── model.py
```

## Achievement

I simply transfer some useful scripts from my upper stream repo `NNAF` (https://github.com/AmadFat/NNAF) to make the development really fast. Although this program can use `NNAF` as a dependency, I mean to give up this because `NNAF` is currently under extreme evolution to enhance the practical performance.

## Acknowledgement

If you like these codes, plz leave a star, thank you!