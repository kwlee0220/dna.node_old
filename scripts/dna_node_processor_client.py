import json

from dna import config
from dna.execution import ExecutionState

from dna.node.pika.pika_execution import PikaConnector
from dna.node.pika.pika_execution_client import PikaExecutionClient


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Detect Objects in a video file")
    parser.add_argument("req_message", help="request file path")
    parser.add_argument("--host", "-i", metavar="broker host", help="RabbitMQ broker host", default="localhost")
    parser.add_argument("--port", "-p", metavar="broker port", help="RabbitMQ broker port", default=5672)
    parser.add_argument("--user", "-u", metavar="broker user", help="RabbitMQ broker user id", default="dna")
    parser.add_argument("--password", "-w", metavar="broker password", help="RabbitMQ broker password",
                        default="urc2004")
    parser.add_argument("--request_qname",  "-q", help="request queue name", default='track_requests')
    parser.add_argument("--servers", help="bootstrap-servers", default='localhost:9091,localhost:9092,localhost:9093')
    return parser.parse_known_args()


def main():
    args, unknown = parse_args()
    
    json_str = ''
    with open(args.req_message, 'r') as file:
        json_str = file.read()
        
    connector = PikaConnector(host=args.host, port=args.port, user_id=args.user, password=args.password)
    client = PikaExecutionClient(connector=connector, request_qname=args.request_qname)
    try:
        client.start(json_str)
        for resp in client.report_progress():
            resp = config.to_conf(resp)
            print(resp)
            # if resp.state == 'RUNNING' and resp.progress.frame_index >= 40:
            #     client.stop(resp.id, 'timeout')
    finally:
        client.close()

if __name__ == '__main__':
	main()