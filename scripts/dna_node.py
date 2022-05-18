from threading import Thread

from datetime import timedelta

from omegaconf import OmegaConf
from dna import track
from pubsub import PubSub

import dna
from dna.camera import Camera, ImageProcessor, create_image_processor
from dna.node import TrackEventSource, RefineTrackEvent, KafkaEventPublisher, \
                    DropShortTrail, GenerateLocalPath, PrintTrackEvent, EventPublisher
from dna.enhancer.world_transform import WorldTransform
from dna.track import load_object_tracking_callback


pubsub = PubSub()
PUB_TRACK_EVENT = EventPublisher(pubsub, "track_events")
PUB_REFINED = EventPublisher(pubsub, "track_events_refined")
PUB_LONG_TRAILS = EventPublisher(pubsub, "long_trails")
PUB_WORLD_COORD = EventPublisher(pubsub, "track_events_world_coord")
PUB_LOCAL_PATHS = EventPublisher(pubsub, "local_paths")


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Detect Objects in a video file")
    parser.add_argument("conf_path", help="configuration file path")
    # parser.add_argument("--track_file", help="Object track log file.", default=None)
    # parser.add_argument("--tracker", help="tracker", default="tracker.deep_sort")
    return parser.parse_known_args()

def main():
    args, unknown = parse_args()

    conf = dna.load_config(args.conf_path)

    camera_params = Camera.Parameters.from_conf(conf.camera)
    camera = dna.camera.create_camera(camera_params)

    source = TrackEventSource(conf.node.id, PUB_TRACK_EVENT)

    refine = RefineTrackEvent(source.subscribe(), PUB_REFINED)
    Thread(target=refine.run, args=tuple()).start()

    queue = refine.subscribe()
    min_trail_length = dna.conf.get_config_value(conf.node, 'min_trail_length').getOrElse(0)
    if min_trail_length > 0:
        drop_short_trail = DropShortTrail(queue, PUB_LONG_TRAILS, conf.node.min_trail_length)
        Thread(target=drop_short_trail.run, args=tuple()).start()
        queue = drop_short_trail.subscribe()

    world_coords = WorldTransform(queue, PUB_WORLD_COORD, conf.camera_geometry)
    Thread(target=world_coords.run, args=tuple()).start()

    te_topic = dna.conf.get_config_value(conf.node.kafka.topics, 'track_events').getOrNone()
    if te_topic is not None:
        tk_pub = KafkaEventPublisher(world_coords.subscribe(), te_topic, conf.node.kafka)
        Thread(target=tk_pub.run, args=tuple()).start()

    lpe_topic = dna.conf.get_config_value(conf.node.kafka.topics, 'local_path_events').getOrNone()
    if lpe_topic is not None:
        gen_lp = GenerateLocalPath(world_coords.subscribe(), PUB_LOCAL_PATHS, conf.node)
        Thread(target=gen_lp.run, args=tuple()).start()

        lp_pub = KafkaEventPublisher(gen_lp.subscribe(), lpe_topic, conf.node.kafka)
        Thread(target=lp_pub.run, args=tuple()).start()

    track_output = dna.conf.get_config_value(conf.node, 'output').getOrNone()
    if track_output is not None:
        print_event = PrintTrackEvent(world_coords.subscribe(), track_output)
        Thread(target=print_event.run, args=tuple()).start()

    show = dna.get_config_value(conf.node, 'show').getOrElse(False)
    if show:
        conf.node.window_name = f'id={conf.node.id}, camera={conf.camera.uri}'

    domain = dna.Box.from_size(camera.parameters.size)
    output_video = dna.get_config_value(conf.node, 'output_video').getOrNone()
    cb = load_object_tracking_callback(conf.tracker, domain, show, output_video, [source])

    proc_params = ImageProcessor.Parameters.from_conf(conf.node)
    proc = create_image_processor(params=proc_params, capture=camera.open(), callback=cb)
    elapsed, frame_count, fps_measured = proc.run()

    print(f"elapsed_time={timedelta(seconds=elapsed)}, frame_count={frame_count}, fps={fps_measured:.1f}" )

if __name__ == '__main__':
	main()