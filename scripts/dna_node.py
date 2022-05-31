from threading import Thread
from datetime import timedelta

from omegaconf import OmegaConf

import dna
from dna.camera import Camera, ImageProcessor, create_image_processor
from dna.node import TrackEventSource, RefineTrackEvent, DropShortTrail, KafkaEventPublisher, \
                    GenerateLocalPath, PrintTrackEvent, EventQueue
from dna.enhancer.world_transform import WorldTransform
from dna.track.utils import load_object_tracking_callback


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Detect Objects in a video file")
    parser.add_argument("conf_path", help="configuration file path")
    # parser.add_argument("--track_file", help="Object track log file.", default=None)
    parser.add_argument("--output", "-o", metavar="json file", help="track event file.", default=None)
    parser.add_argument("--output_video", "-v", metavar="mp4 file", help="output video file.", default=None)
    parser.add_argument("--show", "-s", action='store_true')
    parser.add_argument("--show_progress", "-p", help="display progress bar.", action='store_true')
    return parser.parse_known_args()

def build_pipeline(in_queue: EventQueue, pipe_conf: OmegaConf) -> EventQueue:
    queue0 = in_queue

    if pipe_conf.get('refine_track_events', False):
        refine = RefineTrackEvent()
        queue0.subscribe(refine)
        queue0 = refine

    if pipe_conf.get('drop_short_trails', None) is not None:
        min_trail_length = pipe_conf.drop_short_trails.get('min_length', 0)
        if min_trail_length > 0:
            drop_short_trail = DropShortTrail(min_trail_length)
            queue0.subscribe(drop_short_trail)
            queue0 = drop_short_trail

    if pipe_conf.get('attach_world_coordinates', None) is not None:
        world_coords = WorldTransform(pipe_conf.attach_world_coordinates)
        queue0.subscribe(world_coords)
        queue0 = world_coords

    if pipe_conf.get('publish_to_kafka', None) is not None:
        tk_pub = KafkaEventPublisher(pipe_conf.publish_to_kafka)
        queue0.subscribe(tk_pub)
        top1 = tk_pub

    if dna.conf.get_config(pipe_conf, 'generate_local_paths.kafka', None) is not None:
        gen_lp = GenerateLocalPath(pipe_conf.generate_local_paths)
        queue0.subscribe(gen_lp)
        queue2 = gen_lp

        lp_pub = KafkaEventPublisher(pipe_conf.generate_local_paths.kafka)
        queue2.subscribe(lp_pub)
        queue2 = lp_pub

    return queue0

def main():
    args, unknown = parse_args()
    conf:OmegaConf = dna.load_config(args.conf_path)

    camera:Camera = dna.camera.create_camera(conf.camera)
    proc:ImageProcessor = dna.camera.create_image_processor(camera, OmegaConf.create(vars(args)))

    source = TrackEventSource(conf.id)
    if conf.get('pipeline', None) is not None:
        queue = build_pipeline(source, conf.pipeline)
        if args.output is not None:
            queue.subscribe(PrintTrackEvent(args.output))
    proc.callback = load_object_tracking_callback(camera, proc, conf.tracker, tracker_callbacks=[source])

    elapsed, frame_count, fps_measured = proc.run()
    print(f"elapsed_time={timedelta(seconds=elapsed)}, frame_count={frame_count}, fps={fps_measured:.1f}" )

if __name__ == '__main__':
	main()