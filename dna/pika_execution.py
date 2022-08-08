from typing import Callable
from abc import ABCMeta, abstractmethod

from datetime import timedelta, datetime, time
from omegaconf import OmegaConf
import pika
import uuid

import dna
from .execution import ExecutionContext, ExecutionState, CancellationError, Execution, AsyncExecution
from .pika_rpc import Serde, RpcCallError, JSON_SERDE

import logging
LOGGER = logging.getLogger('dna.pika_execution')

class PikaExecutionContext(ExecutionContext):
    def __init__(self, request:object, channel, method, reply_to:str, properties:pika.BasicProperties) -> None:
        self.request = request
        self.channel = channel
        self.method = method
        self.reply_to = reply_to
        self.props = properties
        self.response_props = pika.BasicProperties(correlation_id=self.props.correlation_id)
        self.serde = JSON_SERDE

        self.control_qname = None

    def started(self) -> None:
        self.control_qname = self.channel.queue_declare(queue='', exclusive=True).method.queue
        started = {
            'id': self.request['id'],
            'state': 'STARTED',
            'timestamp': dna.utils.utc_now(),
            'control_queue': self.control_qname
        }
        self.reply(started)
        self.channel.basic_ack(delivery_tag=self.method.delivery_tag)

    def report_progress(self, progress:object) -> None:
        progress = {
            'id': self.request['id'],
            'state': ExecutionState.RUNNING.name,
            'timestamp': dna.utils.utc_now(),
            'progress': progress
        }
        self.reply(progress)

    def completed(self, result:object) -> None:
        completed = {
            'id': self.request['id'],
            'state': ExecutionState.COMPLETED.name,
            'timestamp': dna.utils.utc_now(),
            'result': result
        }
        self.reply(completed)
        self.channel.queue_delete(queue=self.control_qname)

    def stopped(self, details:str) -> None:
        failed = {
            'id': self.request['id'],
            'state': ExecutionState.STOPPED.name,
            'timestamp': dna.utils.utc_now(),
            'cause': details
        }
        self.reply(failed)
        self.channel.queue_delete(queue=self.control_qname)

    def failed(self, cause:str) -> None:
        failed = {
            'id': self.request['id'],
            'state': ExecutionState.FAILED.name,
            'timestamp': dna.utils.utc_now(),
            'cause': cause
        }
        self.reply(failed)
        self.channel.queue_delete(queue=self.control_qname)

    def reply(self, data:object) -> None:
        self.channel.basic_publish(exchange='', routing_key=self.reply_to, properties=self.response_props,
                                    body=self.serde.serialize(data))


_REQ_QUEUE = 'long_rpc_requests'

def PikaConnectionParameters(host:str='localhost', port:int=5672, user_id='dna', password='') -> pika.ConnectionParameters:
    credentials = pika.PlainCredentials(username=user_id, password=password)
    return pika.ConnectionParameters(host=host, port=port, credentials=credentials)

class PikaExecutionFactory(metaclass=ABCMeta):
    @abstractmethod
    def create(pika_context: PikaExecutionContext) -> Execution: pass
    
class PikaExecutionServer:
    def __init__(self, conn_params: pika.ConnectionParameters,
                 execution_factory: PikaExecutionFactory, request_qname:str=_REQ_QUEUE) -> None:
        self.conn_params = conn_params
        self.req_qname = request_qname
        self.exec_factory = execution_factory
        self.serde = JSON_SERDE
        self.logger = LOGGER

    def run(self) -> None:
        self.conn = pika.BlockingConnection(self.conn_params)

        self.channel = self.conn.channel()
        self.channel.queue_declare(queue=self.req_qname)
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue=self.req_qname, on_message_callback=self.on_request)
        self.channel.start_consuming()

    def on_request(self, channel, method, props:pika.BasicProperties, body:object) -> None:
        try:
            req = self.serde.deserialize(body)
            req = OmegaConf.create(req)
            
            pika_ctx = PikaExecutionContext(req, channel, method, props.reply_to, props)
            work = self.exec_factory.create(pika_ctx)
            result = work.run()
        except Exception as e:
            self.logger.error(f'fails to create execution', e)

class PikaExecutionClient:
    def __init__(self, conn_params:pika.ConnectionParameters, request_qname:str=_REQ_QUEUE,
                progress_handler:Callable[[object],None]=None) -> None:
        self.req_qname = request_qname
        self.progress_handler = progress_handler
        self.serde = JSON_SERDE

        self.conn = pika.BlockingConnection(conn_params)
        self.channel = self.conn.channel()

        self.callback_qname = self.channel.queue_declare(queue='', exclusive=True).method.queue
        self.channel.basic_consume(queue=self.callback_qname, on_message_callback=self.on_response, auto_ack=True)
        self.channel.queue_declare(queue=self.req_qname)

        self.response = None
        self.corr_id = None

    def close(self):
        try:
            self.conn.close()
        finally:
            self.conn = None

    def call(self, request:object) -> object:
        if self.conn is None:
            raise AssertionError(f"{__name__} is closed already")
        
        self.corr_id = str(uuid.uuid4())
        props = pika.BasicProperties(reply_to=self.callback_qname, correlation_id=self.corr_id)
        self.channel.basic_publish(exchange='', routing_key=self.req_qname, properties=props,
                                    body=self.serde.serialize(request))

        while True:
            self.conn.process_data_events(time_limit=None)
            state = self.response['state']
            if state == 'STARTED' or state == ExecutionState.RUNNING.name:
                if self.progress_handler is not None:
                    from contextlib import suppress
                    with suppress(Exception):
                        self.progress_handler(self.response)
            elif state == ExecutionState.COMPLETED.name:
                return self.response['result']
            elif state == ExecutionState.STOPPED.name:
                msg = self.response['cause']
                raise CancellationError(msg)
            elif state == ExecutionState.FAILED.name:
                cause = self.response['cause']
                raise RpcCallError(f'rpc call failed: cause={cause}')
            else:
                raise AssertionError(f'unexpected status message: {state}')

    def on_response(self, channel, method, props:pika.BasicProperties, body:object):
        if props.correlation_id in self.corr_id:
            self.response = self.serde.deserialize(body)