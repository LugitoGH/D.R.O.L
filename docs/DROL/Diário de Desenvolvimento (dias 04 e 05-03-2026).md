## Arquivo Sumiu

O sistema afirmava que estava salvando o sinal. O console mostrava sucesso. Mas o `sinais.json` não aparecia no Windows.

Foi necessário entrar no container e investigar. O arquivo estava lá — dentro de `/app/DROL/data`.

O problema não era código. Era volume mal configurado.

Ao montar corretamente o diretório `DROL/data` do host para `/app/DROL/data` no container, a persistência passou a funcionar como esperado.

Essa etapa foi fundamental para entender, de forma concreta, como o Docker isola o filesystem e como a ponte com o host precisa ser declarada explicitamente.

Foi um aprendizado estrutural, não apenas técnico.

---
## Coluna Vertebral

Houve um ponto claro em que ficou evidente que o DROL precisava parar de ser um conjunto de testes espalhados. Scripts separados resolviam problemas pontuais, mas não conversavam entre si de forma estruturada. Estava confuso.

Decidi concentrar tudo em `DROL/registrar_sinal.py`.

Captura, registro, reconhecimento, interface web, controle de estado, logs — tudo passou a morar no mesmo núcleo. O projeto está crescendo e precisamos nos organizar.

O DROL ganhou espinha dorsal.

---

## O Incômodo do IP Manual

Trabalhar em equipe trouxe um problema prático e irritante: cada notebook tem um IP diferente. Toda vez que alguém mudava de máquina, era necessário alterar o endereço do stream da câmera no código. Algo que, para os desenvolvedores mirins, pode estagnar por não saberem localizar esse problema.

Decidi tentar uma solução usando variáveis de ambiente. Primeiro priorizando `CAMERA_STREAM_URL`. Se ela estiver definida, o backend simplesmente usa. Se não estiver, ele monta a URL com host, porta e caminho específicos. Se nada for informado, tenta `host.docker.internal`. Em Linux, lê o gateway direto de `/proc/net/route`.

O sistema passou a se adaptar ao ambiente em vez de exigir ajustes manuais.

Foi um daqueles momentos silenciosos em que você percebe que o projeto amadureceu.

---

## O Stream Caiu

Em algum teste, o stream simplesmente falhou. E o backend fez o que todo protótipo faz: travou.

Inaceitável.

Implementei um mecanismo de reconexão automática. Quando a leitura do frame falha, o erro aparece claramente no console e o sistema tenta restabelecer a conexão sem derrubar o servidor.

Os logs começaram a exibirr de forma simples o que estava acontecendo internamente: "Queda detectada", "Tentativa de reconexão", "Stream restabelecido".

Esse método me tranquiliza. Não é mais um sistema que quebra ao primeiro imprevisto.

---

## Tornando o Console Útil

Conforme o projeto crescia, o terminal começou a ficar confuso. Faltavam mensagens claras sobre o modo atual, falhas ou instruções de uso.

Os logs foram reorganizados. Mudanças de modo agora aparecem explicitamente. No startup, o backend explica como utilizar o sistema. Falhas no stream são descritas com clareza.

Isso não aparece na interface, apenas na IDE/terminal.

---

## Uma Interface que Finalmente Faz Sentido

Criei, de forma crua mesmo, uma interface que unifica as ferramentas principais. Que alívio tudo ter dado certo de primeira! Nesta interface temos botões como "Registrar sinal", "Reconhecer sinal" e um input para dar nome ao sinal.

Achei interessante implementar também uma vizualização dos logs de registro e reconhecimento para não precisar ir e voltar para a IDE.

![[WhatsApp Image 2026-03-05 at 10.22.55.jpeg|359]]

Além disso adicionei legenda no reconhecimento:

![[WhatsApp Image 2026-03-05 at 10.24.39.jpeg|359]]

---

## O Reconhecimento Ficou Mais Inteligente

Os primeiros testes usavam coordenadas simples. Funcionavam, mas eram frágeis por conta de verificarem exatamente as coordenadas x,y,z no espaço.

Decidi então, realizar uma mudança para vetores normalizados. Cada sinal é tratado como um ponto em um espaço matemático de 63 dimensões. O reconhecimento se baseia na comparação geométrica entre a mão atual e os vetores salvos.

A menor distância encontrada determina o candidato. Um limiar define se o reconhecimento é confiável.

Não é aprendizado de máquina formal ainda, mas já é raciocínio geométrico consistente.

O DROL começou a pensar em termos de forma, não apenas posição.

#### Explicação matemática:

Inicialmente, comparávamos coordenadas absolutas.

Isso significa que se a mão estivesse 10 pixels deslocada, a diferença já poderia invalidar o reconhecimento.

Agora cada mão é representada por um vetor:

$$  
\vec{v} \in \mathbb{R}^{63}  
$$

$$  
21 \times 3 = 63  
$$

Para comparar dois sinais:

$$  
d(\vec{a}, \vec{b}) =  
\sqrt{  
\frac{1}{63}  
\sum_{i=1}^{63}  
(a_i - b_i)^2  
}  
$$
Essa é a distância euclidiana média.

Se:

$$  
d(\vec{a}, \vec{b}) < \text{threshold}  
$$

consideramos reconhecido.

Isso transforma o problema em comparação geométrica.

Cada sinal vira um ponto em um espaço de 63 dimensões. O reconhecimento é encontrar o ponto mais próximo.

Ainda não é um modelo treinado, mas já é raciocínio matemático consistente.

---
## Validação e Estabilidade

O backend consolidado foi validado com `py_compile`. Nenhum erro de sintaxe.

O sistema agora possui:

Persistência funcional  
Stream resiliente  
Interface integrada  
Reconhecimento contínuo  
Configuração dinâmica  
Logs organizados

---

## O Estado Atual

Hoje o DROL possui arquitetura clara, responsabilidades bem definidas e comportamento previsível.

Ainda há muito para evoluir. A comparação direta pode se tornar múltiplas amostras por sinal. O reconhecimento pode migrar para um classificador real. A separação de camadas pode ficar ainda mais elegante.

Podemos permitir múltiplas amostras por sinal e calcular média vetorial no futuro com:

$$  
\vec{m} = \frac{1}{n} \sum_{k=1}^{n} \vec{v_k}  
$$

Continuo trabalhando para termos uma base firme.