
from omegaconf import OmegaConf

from dna.conf import exists_config
from dna.camera import ImageProcessor, ImageCapture
from dna.execution import Execution, ExecutionContext, NoOpExecutionContext
from dna.pika_execution import PikaExecutionContext, PikaExecutionFactory
from dna.tracker import TrackProcessor


_DEFAULT_EXEC_CONTEXT = NoOpExecutionContext()

def build_node_processor(capture: ImageCapture, conf: OmegaConf,
                         context: ExecutionContext=_DEFAULT_EXEC_CONTEXT) -> ImageProcessor:
    from dna.camera import create_camera_from_conf
    from dna.node.utils import load_publishing_pipeline

    img_proc = ImageProcessor(capture, conf, context=context)

    publishing_conf = conf.get('publishing', OmegaConf.create())
    publish_pipeline:TrackProcessor = load_publishing_pipeline(conf.id, publishing_conf)
    
    from dna.tracker.track_pipeline import TrackingPipeline
    tracker_conf = conf.get('tracker', OmegaConf.create())
    frame_proc = TrackingPipeline.load(img_proc, tracker_conf, [publish_pipeline])
    img_proc.add_frame_processor(frame_proc)
    
    return img_proc
    
class PikaNodeExecutionFactory(PikaExecutionFactory):
    def __init__(self, db_conf: OmegaConf, show: bool) -> None:
        super().__init__()
        self.db_conf = db_conf
        self.show = show

    def create(self, pika_ctx: PikaExecutionContext) -> Execution:
        request = OmegaConf.create(pika_ctx.request)

        if exists_config(request, 'node'):
            from .utils import read_node_config
            conf = read_node_config(self.db_conf, request.node)
        elif exists_config(request, 'parameters'):
            conf = request.parameters
            conf.id = request.id
        else:
            raise ValueError(f'cannot get node configuration: request={request}')
        
        from dna.camera.utils import create_camera_from_conf
        camera = create_camera_from_conf(conf.camera)
        if self.show and conf.get('window_name', None) is None:
            conf.window_name = f'camera={camera.uri}'
        
        import dna
        img_proc = build_node_processor(camera.open(), conf, context=pika_ctx)
        if dna.conf.exists_config(request, 'progress_report.interval_seconds'):
            interval = int(request.progress_report.interval_seconds)
            img_proc.report_interval = interval

        return img_proc