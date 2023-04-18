protoc -I=. --python_out=. reid_metrics.proto
python -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. node_processor.proto