
import heapq
import time

from dna import Box
from dna.utils import initialize_logger
from dna.node.publish_events_execution import PikaEventPublisherFactory
from dna.pika_execution import PikaExecutionServer, PikaConnectionParameters


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
    parser.add_argument("--servers", help="bootstrap-servers", default='localhost:9091,localhost:9092,localhost:9093')
    parser.add_argument("--topic", help="topic name", default='node-tracks')
    parser.add_argument("--sync", help="sync to publish events", action='store_true')
    return parser.parse_known_args()

def main():
    initialize_logger()

    args, _ = parse_args()
    conn_params = PikaConnectionParameters(host=args.host, port=args.port,
                                            user_id=args.user, password=args.password)
    fact = PikaEventPublisherFactory(args.topic, args.servers.split(','), args.sync)
    server = PikaExecutionServer(conn_params=conn_params, execution_factory=fact,
                                 request_qname=args.request_qname)
    server.run()

if __name__ == '__main__':
	main()