
from omegaconf import OmegaConf

import dna
from dna.camera import Camera, ImageProcessor
from dna.camera.utils import create_camera_from_conf


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Display a video")
    parser.add_argument("conf_path", help="configuration file path")
    parser.add_argument("--show", "-s", action='store_true')
    parser.add_argument("--show_progress", "-p", help="display progress bar.", action='store_true')
    return parser.parse_known_args()

def main():
    args, _ = parse_args()

    dna.initialize_logger()
    conf = dna.load_config(args.conf_path)
    dna.conf.update(conf, vars(args), ['show', 'show_progress'])

    if conf.get('show', False) and conf.get('window_name', None) is None:
        conf.window_name = f'camera={conf.camera.uri}'

    camera:Camera = create_camera_from_conf(conf.camera)
    img_proc = ImageProcessor(camera.open(), conf)
    result = img_proc.run()
    print(result)

if __name__ == '__main__':
	main()