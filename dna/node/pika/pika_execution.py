from __future__ import annotations

from typing import Optional
from dataclasses import dataclass, field

from argparse import Namespace
from omegaconf import OmegaConf
import pika
from pathlib import Path
import logging

from dna import config, Frame
from dna.camera import ImageProcessor, FrameProcessor, create_opencv_camera_from_conf
from .pika_execution_context import PikaExecutionContext

LOGGER = logging.getLogger('dna.node.pika')


@dataclass(frozen=True) # slots=True
class PikaConnector:
    password:str
    host:str = field(default='localhost')
    port:int = field(default=5672)
    user_id:str = field(default='dna')
    
    def blocking_connection(self) -> pika.BlockingConnection:
        try:
            credentials = pika.PlainCredentials(username=self.user_id, password=self.password)
            conn_params = pika.ConnectionParameters(host=self.host, port=self.port, credentials=credentials)
            return pika.BlockingConnection(conn_params)
        except pika.exceptions.AMQPConnectionError as e:
            import sys
            print(f"fails to connect RabbitMQ broker: host={self.host}, port={self.port}, "
                  f"username={self.user_id}, password={self.password}", file=sys.stderr)
            raise e


class PikaExecutionFactory:
    def __init__(self, args:Namespace) -> None:
        super().__init__()
        self.args = args

    def create(self, pika_ctx:PikaExecutionContext) -> ImageProcessor:
        from dna.node.node_processor import build_node_processor

        request = OmegaConf.create(pika_ctx.request)
        if config.exists(request, 'node'):
            conf_root = Path(self.args.conf_root)
            node_conf_fname = request.node.replace(":", "_") + '.yaml'
            path = conf_root / node_conf_fname
            try:
                conf = config.load(path)
            except Exception as e:
                LOGGER.error(f"fails to load node configuration file: conf_root={self.args.conf_root}, node={request.node}, path='{path}'")
                raise e
                
        # args에 포함된 ImageProcess 설정 정보를 추가한다.
        config.update_values(conf, self.args, 'show', 'show_progress')
        if self.args.kafka_brokers:
            # 'kafka_brokers'가 설정된 경우 publishing 작업에서 이 broker로 접속하도록 설정한다.
            config.update(conf, 'publishing.plugins.kafka_brokers', self.args.kafka_brokers)
        config.update_values(conf, request, "show")
            
        config.update(conf, "camera", request.camera)
        camera = create_opencv_camera_from_conf(conf.camera)
        
        options = config.filter(conf, 'show', 'show_progress')
        img_proc = ImageProcessor(camera.open(), context=pika_ctx, **options)
        
        build_node_processor(img_proc, conf)
        
        report_interval = config.get(request, 'report_frame_interval', default=-1)
        if report_interval > 0:
            img_proc.add_frame_processor(PikaExecutionProgressReporter(pika_ctx, report_interval))

        return img_proc


class PikaExecutionProgressReporter(FrameProcessor):
    def __init__(self, context:PikaExecutionContext, report_frame_interval:int) -> None:
        super().__init__()
        self.ctx = context
        self.report_frame_interval = report_frame_interval

    def on_started(self, proc:ImageProcessor) -> None:
        self.next_report_frame = self.report_frame_interval

    def on_stopped(self) -> None: pass

    def process_frame(self, frame:Frame) -> Optional[Frame]:
        if frame.index >= self.next_report_frame:
            progress = {
                'frame_index': frame.index
            }
            self.ctx.report_progress(progress)
            self.next_report_frame += self.report_frame_interval
        return frame