import base64

message = "Hello, World!"
message_bytes = message.encode('utf-8')
base64_bytes = base64.b64encode(message_bytes)
base64_message = base64_bytes.decode('utf-8')

print(base64_message)

decoded_bytes = base64.b64decode(base64_bytes)
decoded_message = decoded_bytes.decode('utf-8')

print(decoded_message)