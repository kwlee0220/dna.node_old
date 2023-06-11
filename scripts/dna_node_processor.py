
import logging
import argparse
from omegaconf import OmegaConf

from dna import config, initialize_logger
from dna.node.pika.pika_execution import PikaExecutionFactory, PikaConnector
from dna.node.pika.pika_execution_server import PikaExecutionServer
from scripts import update_namespace_with_environ

LOGGER = logging.getLogger('dna.node.pika')


def parse_args():
    parser = argparse.ArgumentParser(description="Run a node processor.")
    parser.add_argument("--conf_root", metavar="dir", help="Root directory for configurations", default="conf")
    parser.add_argument("--kafka_brokers", nargs='+', metavar="hosts", help="Kafka broker hosts list", default=None)
    parser.add_argument("--rabbitmq_host", metavar="broker host", help="RabbitMQ broker host", default="localhost:5672")
    parser.add_argument("--rabbitmq_user", metavar="broker user", help="RabbitMQ broker user id", default="dna")
    parser.add_argument("--rabbitmq_password", metavar="broker password", help="RabbitMQ broker password",
                        default="urc2004")
    parser.add_argument("--request_qname", "-q", metavar="json file", help="track event file.",
                        default="track_requests")
    parser.add_argument("--show", "-s", nargs='?', const='0x0', default='0x0')

    parser.add_argument("--logger", metavar="file path", help="logger configuration file path")
    return parser.parse_known_args()

def main():
    args, _ = parse_args()
    initialize_logger(args.logger)
    args = update_namespace_with_environ(args)
    
    host, port = tuple(args.rabbitmq_host.split(':'))
    connector = PikaConnector(host=host, port=int(port),
                              user_id=args.rabbitmq_user, password=args.rabbitmq_password)
    fact = PikaExecutionFactory(args=args)
    server = PikaExecutionServer(connector=connector,
                                 execution_factory=fact,
                                 request_qname=args.request_qname,
                                 logger=LOGGER)
    server.run()

if __name__ == '__main__':
	main()