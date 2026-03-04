## Persistência de dados no Docker

Hoje enfrentamos um problema clássico de quem começa a estruturar aplicações com Docker: o arquivo estava sendo salvo, mas não aparecia no Windows.

Os logs mostravam claramente:

```
Sinal capturado!
Sinal 'A' salvo com sucesso.
```

Porém, a pasta `data/` não existia no host.

Após investigar dentro do container usando:

```
docker exec -it <id> sh
pwd
ls
```

descobrimos que o arquivo estava sendo criado em:

```
/app/DROL/data/sinais.json
```

O erro não era no código.  
Era um desalinhamento entre o caminho real dentro do container, o caminho montado via volume e a expectativa de onde o arquivo deveria aparecer.

A solução foi montar corretamente o volume:

```powershell
docker run -p 5000:5000 -v "C:\Users\Aluno\Documents\GitHub\D.R.O.L\DROL\data:/app/DROL/data" drol
```

Com isso, o `sinais.json` passou a persistir fora do container.

---

## Organização de Caminhos no Ambiente Linux

Percebemos também que o projeto roda dentro de:

```
/app
   └── DROL
```

Isso impacta diretamente qualquer uso de:

```python
os.path.dirname(os.path.abspath(__file__))
```

Para evitar ambiguidade futura, ficou definido que caminhos dentro do container devem ser previsíveis, que os volumes precisam casar exatamente com o `WORKDIR` e que o Docker não compartilha filesystem automaticamente. Ele isola por padrão, e qualquer ponte entre host e container precisa ser declarada explicitamente.

Esse entendimento foi essencial para estabilizar o projeto.

---

## Registro de Sinais Funcional

Implementamos o registro contínuo de sinais utilizando captura via MediaPipe Tasks, armazenamento estruturado em JSON e logs claros no terminal para confirmar cada operação.

Sempre que aparece:

```
Sinal capturado!
```

sabemos que as landmarks foram extraídas corretamente e registradas.

Cada nova chamada de:

```
/registrar?nome=A
```

adiciona uma nova amostra ao banco de dados. Isso transformou o sistema de protótipo em um coletor de dataset real. Não estamos mais apenas testando detecção. Estamos construindo base de dados.

---

## Criação do Reconhecimento Contínuo

Após estabilizar o registro, desenvolvemos um novo arquivo responsável exclusivamente pela leitura e identificação dos sinais.

A lógica deixou de ser baseada em evento manual e passou a ser contínua. A cada frame capturado, o sistema detecta as landmarks atuais, percorre o dataset salvo e calcula a distância média entre os pontos da mão atual e cada amostra armazenada.

A menor distância encontrada é selecionada. Se ela estiver abaixo de um limiar definido, o sistema considera que houve reconhecimento válido e exibe no canto superior esquerdo:

```
Sinal: X
```

Caso contrário, exibe:

```
Faça um sinal!
```

Esse ciclo acontece a cada frame. Não depende de rota HTTP, não depende de clique, não depende de comando externo. O reconhecimento virou comportamento permanente do sistema.

O pipeline permanece simples e direto: detecta landmarks, compara com o dataset, seleciona a menor distância, aplica o limiar e exibe o resultado. Sem camadas desnecessárias.

---

## Limitações Estruturais do Docker

Reforçamos uma limitação importante: Docker não acessa webcam do host automaticamente.

Isso nos levou a manter a arquitetura com `camera_server.py` capturando a webcam local, expondo um stream HTTP que é consumido pelo container. O processamento acontece dentro do ambiente Linux isolado, enquanto a captura permanece no sistema hospedeiro.

Essa separação pode parecer mais complexa à primeira vista, mas mantém o projeto portátil e independente do sistema operacional. O container não precisa saber se está rodando em Windows, Linux ou macOS. Ele apenas consome um stream.

---

## Estrutura Atual do Projeto

```
/app
   └── DROL
       ├── registrar_sinal.py
       ├── reconhecer_sinal.py
       ├── data/
       │    └── sinais.json
       ├── models/
       └── tools/
```

Nesse ponto temos coleta de dados funcionando, persistência corretamente configurada, reconhecimento contínuo operando em tempo real, logs estruturados e ambiente Docker estabilizado.

O projeto deixou de ser um experimento frágil e passou a ter estrutura.

---

## Documentação

A documentação segue sendo organizada no [[Obsidian]] para registro técnico offline.

Após consolidação das alterações, os commits serão realizados no [[GitHub]] para versionamento formal do progresso e controle das versões arquiteturais do sistema.

---

## Próximos Passos Técnicos

Hoje estruturamos base, persistência e reconhecimento.

O próximo salto técnico será trabalhar na normalização das landmarks para reduzir dependência de posição absoluta, diminuir falsos positivos com ajustes no limiar e, posteriormente, agrupar amostras por classe para tornar a comparação mais eficiente.

A etapa seguinte será substituir a comparação direta por um classificador real, como SVM ou MLP, permitindo generalização e maior robustez.

O projeto saiu da fase experimental e entrou na fase arquitetural.

Agora ele começa a ficar sério.