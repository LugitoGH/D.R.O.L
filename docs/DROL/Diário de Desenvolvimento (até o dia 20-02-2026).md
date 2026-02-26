# Problemas com bibliotecas

O desenvolvimento do D.R.O.L sofreu com algumas alterações na biblioteca mediapipe que foi atualizada e substituiu algumas funções. 

O código já estava funcional e exibindo, de certa forma, a letra expressada em libras. Porém, após a atualização, nada mais funcionara.

Houve a realização de diversos testes. Com o auxílio da Inteligência Artificial "ChatGPT" da empresa OpenAI, testamos linha por linha, módulo por módulo e não encontramos o real problema, até o momento em que por meio de pesquisa externa chegamos em um post no Github de um usuário com a mesma mensagem de erro: AttributeError: module 'mediapipe' has no attribute 'solutions' e ModuleNotFoundError: No module named 'mediapipe'  que nos trouxe uma luz: "O mediapipe não usa mais solutions, tente agora mediapipe tasks"

---
### Imagens do Post:

![[Pasted image 20260220094108.png]]![[Pasted image 20260220094150.png]]![[Pasted image 20260220094214.png]]

---
User francês nos ajudou muito:

![[Pasted image 20260220094315.png]]
 
Versão Traduzida da imagem acima:
![[Pasted image 20260220094259.png]]

---

# Leitura de documentações e problemas de ambiente

Após a leitura da documentação do Mediapipe Tasks (https://ai.google.dev/edge/mediapipe/solutions/tasks?hl=pt-br) e suporte da IA (que até então não está atualizada com essa versão do mediapipe, porém sua lógica foi capaz de nos ajudar em alguns momentos oportunos) conseguimos voltar ao momento em que estávamos, a captura da mão e seus pontos!

![[Pasted image 20260220095547.png]] Imagem do commit sobre a atualização da bilbioteca no dia 11/02

Como nem tudo são flores, chegou o momento "mas no meu pc tá funcionando" e lutamos para replicar a configuração, mas falhamos completamente. Então chegou o momento de apresentar o [[Docker]] aos garotos!

# Uso de Docker

O [[Docker]] aqui nos será muito útil pois uma de suas principais funções é clonar o ambiente de desenvolvimento e replicar em outras máquinas. Assim o docker nos ajuda na preparação do ambiente e podemos focar no desenvolvimento sem preocupações.

![[Pasted image 20260220100553.png]]
Imagem do commit sobre o uso de Docker no dia 18/02

