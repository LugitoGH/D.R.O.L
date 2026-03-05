# Arquitetura DROL

1. camera_server.py
   - Captura webcam local
   - Disponibiliza stream HTTP

2. Container Docker
   - Consome stream HTTP
   - Executa MediaPipe Tasks
   - Processa landmarks
   - Serve vídeo via Flask

3. Persistência
   - sinais.json salvo em /app/DROL/data
   - Volume montado para host Windows

Fluxo:
Camera → HTTP Stream → Docker → MediaPipe → Comparação → Flask → Navegador