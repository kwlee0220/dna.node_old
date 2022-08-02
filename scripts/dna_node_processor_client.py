import json
from dna.execution import ExecutionState

from dna.pika_execution import PikaExecutionClient, PikaConnectionParameters


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
        
    conn_params = PikaConnectionParameters(host=args.host, port=args.port,
                                            user_id=args.user, password=args.password)
    client = PikaExecutionClient(conn_params=conn_params, request_qname=args.request_qname,
                                progress_handler=lambda x: print(x))
    result = client.call(json_str)
    print("done:", result)

if __name__ == '__main__':
	main()