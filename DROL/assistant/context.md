# DROL - Contexto Atual

Objetivo:
Reconhecimento de sinais de Libras em tempo real usando MediaPipe Tasks.

Status:
- Registro funcional
- Persistência via volume Docker
- Reconhecimento contínuo implementado
- Arquitetura distribuída (camera_server + container)

Problemas conhecidos:
- Dependência de threshold fixo
- Sem normalização de landmarks
- Possível instabilidade de stream HTTP