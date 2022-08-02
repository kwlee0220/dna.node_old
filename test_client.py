import json

from dna.pika_execution import PikaExecutionClient, PikaConnectionParameters


json_str = ''
with open('data/on-demand-requests/req01_etri_05.json', 'r') as file:
# with open('data/on-demand-requests/req01_etri_05.json', 'r') as file:
    json_str = file.read()

params = PikaConnectionParameters(user_id='dna', password='urc2004')
client = PikaExecutionClient(conn_params=params, request_qname='publish_requests', progress_handler=lambda x: print(x))
result = client.call(json_str)

print("done:", result)