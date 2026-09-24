import sys
import os

import ecal.core.core as ecal_core
from ecal.core.subscriber import ProtoSubscriber

try:
    from messages import mi_mensaje_pb2
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import mi_mensaje_pb2

ecal_core.initialize("Python Protobuf Subscriber")
sub = ProtoSubscriber("mensaje 1", mi_mensaje_pb2.HelloWorld)

print("Nivel 0 - Receptor de prueba activo. Esperando mensajes...")

while ecal_core.ok():
    isReceived, protobuf_message, _ = sub.receive(100)
    if isReceived:
        print(f"name={protobuf_message.name} | id={protobuf_message.id} | "
              f"msg={protobuf_message.msg} | state={protobuf_message.state}")

ecal_core.finalize()
