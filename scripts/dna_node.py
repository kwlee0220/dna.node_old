from threading import Thread
from datetime import timedelta

from omegaconf import OmegaConf

import dna
from dna.camera import Camera, ImageProcessor, create_image_processor
from dna.node import TrackEventSource, RefineTrackEvent, DropShortTrail, KafkaEventPublisher, \
                    GenerateLocalPath, PrintTrackEvent, EventQueue
from dna.enhancer.world_transform import WorldTransform
from dna.tracker.utils import load_object_tracking_callback


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Detect Objects in a video file")
    parser.add_argument("conf_path", help="configuration file path")
    parser.add_argument("--output", "-o", metavar="json file", help="track event file.", default=None)
    parser.add_argument("--output_video", "-v", metavar="mp4 file", help="output video file.", default=None)
    parser.add_argument("--show", "-s", action='store_true')
    parser.add_argument("--show_progress", "-p", help="display progress bar.", action='store_true')
    return parser.parse_known_args()

_DEFAULT_MIN_PATH_LENGTH=10

def build_pipeline(queue: EventQueue, pipe_conf: OmegaConf) -> EventQueue:
    # drop unnecessary tracks (eg. trailing 'TemporarilyLost' tracks)
    refine = RefineTrackEvent()
    queue.add_listener(refine)
    queue = refine

    # drop too-short tracks of an object
    min_path_length = pipe_conf.get('min_path_length', _DEFAULT_MIN_PATH_LENGTH)
    if min_path_length > 0:
        drop_short_path = DropShortTrail(min_path_length)
        queue.add_listener(drop_short_path)
        queue = drop_short_path

    # attach world-coordinates to each track
    if pipe_conf.get('attach_world_coordinates') is not None:
        world_coords = WorldTransform(pipe_conf.attach_world_coordinates)
        queue.add_listener(world_coords)
        queue = world_coords

    return queue

def main():
    args, unknown = parse_args()
    conf:OmegaConf = dna.load_config(args.conf_path)

    camera:Camera = dna.camera.create_camera(conf.camera)
    proc:ImageProcessor = dna.camera.create_image_processor(camera, OmegaConf.create(vars(args)))

    source = TrackEventSource(conf.id)
    pipeline_conf = conf.get('pipeline', OmegaConf.create())
    queue = build_pipeline(source, pipeline_conf)
    if conf.get('kafka_publisher', None) is not None:
        queue.add_listener(KafkaEventPublisher(conf.kafka_publisher))
    if args.output is not None:
        queue.add_listener(PrintTrackEvent(args.output))
        
    tracker_conf = conf.get('tracker', OmegaConf.create())
    tracker_conf.output = args.output
    proc.callback = load_object_tracking_callback(camera, proc, tracker_conf, tracker_callbacks=[source])

    elapsed, frame_count, fps_measured = proc.run()
    print(f"elapsed_time={timedelta(seconds=elapsed)}, frame_count={frame_count}, fps={fps_measured:.1f}" )

if __name__ == '__main__':
	main()