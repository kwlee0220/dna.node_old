
from omegaconf import OmegaConf

import dna
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
    parser.add_argument("--show", "-s", nargs='?', const='0x0', default='0x0')

    parser.add_argument("--db_host", metavar="postgresql host", help="PostgreSQL host", default='localhost')
    parser.add_argument("--db_port", metavar="postgresql port", help="PostgreSQL port", default=5432)
    parser.add_argument("--db_name", metavar="dbname", help="PostgreSQL database name", default='dna')
    parser.add_argument("--db_user", metavar="user_name", help="PostgreSQL user name", default='dna')
    parser.add_argument("--db_password", metavar="password", help="PostgreSQL user password", default="urc2004")

    parser.add_argument("--logger", metavar="file path", help="logger configuration file path")
    return parser.parse_known_args()

def main():
    args, _ = parse_args()

    dna.initialize_logger(args.logger)
    conf, db_conf, args_conf = dna.load_node_conf(args)

    conn_params = dna.PikaConnectionParameters(host=args.host, port=args.port,
                                               user_id=args.user, password=args.password)
    server = dna.PikaExecutionServer(conn_params=conn_params,
                                     execution_factory=PikaNodeExecutionFactory(db_conf=db_conf, show=args.show),
                                     request_qname=args.request_qname)
    server.run()

if __name__ == '__main__':
	main()