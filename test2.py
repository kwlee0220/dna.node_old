import uuid

import threading
import time

from dna import Frame
from dna.node.node_processor_grpc import NodeProcessorServicer
from dna.node.proto import node_processor_pb2

service = NodeProcessorServicer()

request = node_processor_pb2.RunNodeProcessRequest(node_id='etri_04',
                                                   sync=True)
for report in service.Run(request, None):
    print(report)