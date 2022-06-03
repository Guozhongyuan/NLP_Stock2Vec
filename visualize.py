'''
    company vector visualization
'''


import numpy as np
from pathlib import Path
import numpy as np
from tqdm import tqdm
import os
from tensorboard.plugins import projector
from pathlib import Path


vectors_path = 'data/vectors_CleanNewsFilter.pkl'

os.makedirs('projector/', exist_ok=True)


'''
    1. Read vectors, build tsv file
'''
data = np.load(vectors_path, allow_pickle=True)
ticker_names = []
ticker_infos = []
vectors = []
for idx in tqdm(range(len(data))):
    if 'vector' in data[idx].keys():
        vectors.append(data[idx]['vector'])
        ticker_infos.append(data[idx]['info'])
        ticker_names.append(data[idx]['ticker'])
vectors = np.stack(vectors)
words = ticker_names
logdir = Path('projector/')
metadata_filename = 'metadata.tsv'
lines = ["ticker_name\tticker_info"]
for name, info in zip(ticker_names, ticker_infos):
    lines.append(f"{name}\t{info}")
logdir.joinpath(metadata_filename).write_text("\n".join(lines), encoding="utf8")
tensor_filename = 'tensor.tsv'
lines = ["\t".join(map(str, vector)) for vector in vectors]
logdir.joinpath(tensor_filename).write_text("\n".join(lines), encoding="utf8")


'''
    2. Build config.pbtxt, projector need it to show
'''
metadata_filename = 'metadata.tsv'
tensor_filename = 'tensor.tsv'
logdir = Path('projector/')

config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.metadata_path = metadata_filename
embedding.tensor_path = tensor_filename
projector.visualize_embeddings(logdir, config)


'''
    3. Run in terminal "tensorboard --logdir=projector/"
'''