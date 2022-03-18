import os
import numpy as np
import pandas as pd


def get_samples(root):
    tsv_file = pd.read_csv(os.path.join(root, 'beauty-icwsm15-dataset.tsv'), sep='\t', header=0)
    samples = {'nature': [], 'animal': [], 'urban': [], 'people': []}
    for photo_id, category, scores in zip(tsv_file['#flickr_photo_id'], tsv_file['category'], tsv_file['beauty_scores']):
        img_path = os.path.join(root, 'img_files', f'{photo_id}.jpg')
        score = int(np.median(np.median([int(s) for s in scores.split(',')])))
        samples[category].append((img_path, score - 1))
    return samples


if __name__ == '__main__':
    get_samples('~/Documents/DataSet/ImageAesthetics')
