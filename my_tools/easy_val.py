import os

config = './configs/subway_transformer/subwayformer_base.py'
ckpt = ''

os.system(f'python tools/val.py {config} {ckpt}')