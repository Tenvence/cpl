import argparse
import os
import xml.etree.ElementTree as et

import pandas as pd
import requests
import tqdm


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_aesthetics_path', default='../../DataSet/ImageAesthetics/beauty-icwsm15-dataset.tsv', type=str)
    parser.add_argument('--output_path', default='../../DataSet/ImageAesthetics/', type=str)
    parser.add_argument('--api_key', default='1188f7f86dd85503a7f28b983a3ca0c5', type=str)
    parser.add_argument('--proxies_port', default=1087, type=int)

    return parser.parse_args()


def main():
    args = get_args_parser()

    xml_path = os.path.join(args.output_path, 'xml_files')
    if not os.path.exists(xml_path):
        os.mkdir(xml_path)

    img_path = os.path.join(args.output_path, 'img_files')
    if not os.path.exists(img_path):
        os.mkdir(img_path)

    proxies = {'http': f'127.0.0.1:{args.proxies_port}', 'https': f'127.0.0.1:{args.proxies_port}'}

    tsv_file = pd.read_csv(args.image_aethetics_path, sep='\t', header=0)
    processor = tqdm.tqdm(tsv_file['#flickr_photo_id'])
    for photo_id in processor:
        if not os.path.exists(os.path.join(xml_path, f'{photo_id}.xml')):
            url = f'https://www.flickr.com/services/rest/?method=flickr.photos.getSizes&api_key={args.api_key}&photo_id={photo_id}&format=rest'
            xml_file = requests.get(url, proxies=proxies)
            open(os.path.join(xml_path, f'{photo_id}.xml'), 'wb').write(xml_file.content)

        if not os.path.exists(os.path.join(img_path, f'{photo_id}.jpg')):
            tree = et.parse(os.path.join(xml_path, f'{photo_id}.xml'))
            root = tree.getroot()
            if root.attrib['stat'] == 'ok':
                img_file = requests.get(root[0][3].attrib['source'], proxies=proxies)
                open(os.path.join(img_path, f'{photo_id}.jpg'), 'wb').write(img_file.content)
            elif root[0].attrib['code'] != '1':
                os.remove(os.path.join(img_path, f'{photo_id}.xml'))
                exit()

        processor.set_description(f'#ok={len(os.listdir(img_path))}; #fail={len(os.listdir(xml_path)) - len(os.listdir(img_path))}')


if __name__ == '__main__':
    main()
