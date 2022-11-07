
from datetime import time, timedelta
from typing import Optional
from abc import ABCMeta, abstractmethod

from enum import Enum
from logging import Logger

class ExecutionState(Enum):
    NOT_STARTED = 0
    STARTING = 1
    RUNNING = 2
    STOPPING = 3
    STOPPED = 4
    FAILED = 5
    COMPLETED = 6

class Execution(metaclass=ABCMeta):
    @abstractmethod
    def run(self) -> object: pass

class ExecutionContext(metaclass=ABCMeta):
    @abstractmethod
    def started(self) -> None: pass
    
    @abstractmethod
    def report_progress(self, progress:object) -> None: pass

    @abstractmethod
    def completed(self, result:object) -> None: pass

    @abstractmethod
    def stopped(self, details:str) -> None: pass

    @abstractmethod
    def failed(self, cause:str) -> None: pass

class NoOpExecutionContext(ExecutionContext):
    def started(self) -> None: pass
    def report_progress(self, progress:object) -> None: pass
    def completed(self, result:object) -> None: pass
    def stopped(self, details:str) -> None: pass
    def failed(self, cause:str) -> None: pass

class LoggingExecutionContext(ExecutionContext):
    def __init__(self, logger:Logger=None) -> None:
        super().__init__()
        self.logger = logger

    def started(self) -> None:
        if self.logger is not None:
            self.logger.info(f'started')

    def report_progress(self, progress:object) -> None: 
        if self.logger is not None:
            self.logger.info(f'progress reported: progress={progress}')

    def completed(self, result:object) -> None:
        if self.logger is not None:
            self.logger.info(f'completed: result={result}')

    def stopped(self, details:str) -> None:
        if self.logger is not None:
            self.logger.info(f'stopped: details={details}')

    def failed(self, cause:str) -> None:
        if self.logger is not None:
            self.logger.info(f'failed: cause={cause}')

class ExecutionFactory(metaclass=ABCMeta):
    @abstractmethod
    def create(self, context:ExecutionContext) -> Execution: pass



class CancellationError(Exception):
    def __init__(self, message:str) -> None:
        self.message = message
        super().__init__(message)
    
    
import threading
class AbstractExecution(Execution):
    def __init__(self, report_interval=60*60, context:ExecutionContext= NoOpExecutionContext()):
        self.ctx = context
        
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.state = ExecutionState.NOT_STARTED
        self.stop_details = None

        self.report_interval = report_interval
        
    def context(self) -> ExecutionContext:
        return self.ctx
        
    @abstractmethod
    def run_work(self) -> object: pass

    @abstractmethod
    def finalize(self) -> None: pass
    
    def check_stopped(self) -> None:
        with self.lock:
            if self.state == ExecutionState.STOPPING:
                raise CancellationError(self.stop_details)

    def run(self) -> object:
        with self.lock:
            if self.state != ExecutionState.NOT_STARTED:
                raise AssertionError(f'invalid execution state: {self.state.name}, expected={ExecutionState.NOT_STARTED.name}')
            self.state = ExecutionState.RUNNING
            self.cond.notify_all()
        self.ctx.started()
        
        try:
            result = self.run_work()
            with self.lock:
                if self.state != ExecutionState.RUNNING and self.state != ExecutionState.STOPPING:
                    raise AssertionError(f'invalid execution state: {self.state.name}, expected={ExecutionState.RUNNING.name}')
                self.state = ExecutionState.COMPLETED
                self.cond.notify_all()
            self.ctx.completed(result)
            
            return result
        except CancellationError as e:
            with self.lock:
                if self.state != ExecutionState.RUNNING and self.state != ExecutionState.STOPPING:
                    raise AssertionError(f'invalid execution state: {self.state.name}, expected={ExecutionState.RUNNING.name}')
                self.state = ExecutionState.STOPPED
                self.cond.notify_all()
            self.ctx.stopped(str(e))
        except Exception as e:
            with self.lock:
                self.state = ExecutionState.FAILED
                self.cond.notify_all()
            self.ctx.failed(str(e))
        finally:
            self.finalize()
        
    def stop(self, details: str='user requested', nowait=False) -> None:
        with self.lock:
            if self.state == ExecutionState.RUNNING:
                self.stop_details = details
                self.state = ExecutionState.STOPPING
                if not nowait:
                    while self.state == ExecutionState.STOPPING:
                        self.cond.wait()


class InvocationError(Exception):
    def __init__(self, message:str) -> None:
        self.message = message
        super().__init__(message)

class TimeoutError(Exception):
    def __init__(self, message:str) -> None:
        self.message = message
        super().__init__(message)

class AsyncExecution(Execution):
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.state = ExecutionState.NOT_STARTED
        self.result = None
        self.message = None

    def is_started(self) -> bool:
        with self.lock: return self.state >= ExecutionState.RUNNING

    def is_running(self) -> bool:
        with self.lock: return self.state == ExecutionState.RUNNING

    def is_completed(self) -> bool:
        with self.lock: return self.state == ExecutionState.COMPLETED

    def is_failed(self) -> bool:
        with self.lock: return self.state == ExecutionState.FAILED

    def is_stopped(self) -> bool:
        with self.lock: return self.state == ExecutionState.STOPPED

    def wait_for_finished(self, timeout:float=timedelta.max.total_seconds()) -> ExecutionState:
        due = time.time() + timeout
        with self.lock:
            while True:
                timeout = due - time.time()
                if self.cond.wait_for(lambda: self.state >= ExecutionState.STOPPED, timeout):
                    return self.state
                else:
                    raise TimeoutError(f"timeout={timedelta(seconds=timeout)}")

    def get(self, timeout:float=timedelta.max.total_seconds()) -> object:
        due = time.time() + timeout
        with self.lock:
            while True:
                timeout = due - time.time()
                if self.cond.wait_for(lambda: self.state >= ExecutionState.STOPPED, timeout):
                    if self.state == ExecutionState.COMPLETED:
                        return self.result
                    elif self.state == ExecutionState.STOPPED:
                        raise CancellationError(self.message)
                    elif self.state == ExecutionState.FAILED:
                        raise InvocationError(self.message)
                    else:
                        raise AssertionError(f"unexpected state: {self.state}")
                else:
                    raise TimeoutError(f"timeout={timedelta(seconds=timeout)}")

    def notify_started(self) -> None:
        with self.lock:
            self.state = ExecutionState.RUNNING
            self.cond.notify_all()

    def notify_completed(self, result: object) -> None:
        with self.lock:
            self.result = result
            self.state = ExecutionState.COMPLETED
            self.cond.notify_all()

    def notify_stopped(self, message: str) -> None:
        with self.lock:
            self.message = message
            self.state = ExecutionState.STOPPED
            self.cond.notify_all()

    def notify_failed(self, message: str) -> None:
        with self.lock:
            self.message = message
            self.state = ExecutionState.FAILED
            self.cond.notify_all()