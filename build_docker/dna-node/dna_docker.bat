@ECHO OFF

@REM localhost에 해당하는 local ip 주소를 환경변수 LOCAL_IP에 저장
for /F "tokens=2 delims=:" %%i in ('"ipconfig | findstr IP | findstr 192."') do SET LOCAL_IP=%%i
set LOCAL_IP=%LOCAL_IP: =%

docker run -it --name dna_node --rm ^
						--gpus all ^
						-e DISPLAY=%LOCAL_IP%:0.0 ^
						--network=host ^
						-v %DNA_NODE_HOME%/conf:/dna.node/conf ^
						-v %DNA_NODE_HOME%/data:/dna.node/data ^
						-v %DNA_NODE_HOME%/regions:/dna.node/regions ^
						-v %DNA_NODE_HOME%/models:/dna.node/models ^
						kwlee0220/dna-node ^
						%*