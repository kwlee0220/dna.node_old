
from omegaconf import OmegaConf

import dna
from dna.camera.image_processor import ImageProcessor
from dna.node.node_processor import PikaNodeExecutionFactory

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Detect Objects in a video file")
    parser.add_argument("--host", "-i", metavar="broker host", help="RabbitMQ broker host", default="localhost")
    parser.add_argument("--port", "-p", metavar="broker port", help="RabbitMQ broker port", default=5672)
    parser.add_argument("--user", "-u", metavar="broker user", help="RabbitMQ broker user id", default="dna")
    parser.add_argument("--password", "-w", metavar="broker password", help="RabbitMQ broker password",
                        default="urc2004")
    parser.add_argument("--request_qname", "-q", metavar="json file", help="track event file.",
                        default="track_requests")
    parser.add_argument("--show", "-s", action='store_true')
    return parser.parse_known_args()

def main():
    dna.initialize_logger()

    args, _ = parse_args()
    conn_params = dna.PikaConnectionParameters(host=args.host, port=args.port,
                                               user_id=args.user, password=args.password)
    server = dna.PikaExecutionServer(conn_params=conn_params,
                                     execution_factory=PikaNodeExecutionFactory(args.show),
                                     request_qname=args.request_qname)
    server.run()

if __name__ == '__main__':
	main()