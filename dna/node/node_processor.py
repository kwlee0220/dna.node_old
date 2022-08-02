
from omegaconf import OmegaConf

from dna.camera import ImageProcessor, ImageCapture
from dna.execution import Execution, ExecutionContext, NoOpExecutionContext
from dna.pika_execution import PikaExecutionContext, PikaExecutionFactory
from dna.tracker import TrackProcessor


_DEFAULT_EXEC_CONTEXT = NoOpExecutionContext()

def build_node_processor(capture: ImageCapture, conf: OmegaConf,
                         context: ExecutionContext=_DEFAULT_EXEC_CONTEXT) -> ImageProcessor:
    from dna.camera.utils import create_camera_from_conf
    from dna.node.utils import load_publishing_pipeline
    from dna.tracker.utils import build_track_pipeline

    img_proc = ImageProcessor(capture, conf, context=context)

    publishing_conf = conf.get('publishing', OmegaConf.create())
    publish_pipeline:TrackProcessor = load_publishing_pipeline(conf.id, publishing_conf)
    
    tracker_conf = conf.get('tracker', OmegaConf.create())
    img_proc.callback = build_track_pipeline(img_proc, tracker_conf, [publish_pipeline])
    
    return img_proc
    
class PikaNodeExecutionFactory(PikaExecutionFactory):
    def __init__(self, show: bool) -> None:
        super().__init__()
        self.show = show

    def create(self, pika_ctx: PikaExecutionContext) -> Execution:
        request = OmegaConf.create(pika_ctx.request)
        conf = request.parameters
        conf.id = pika_ctx.request.id
        
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