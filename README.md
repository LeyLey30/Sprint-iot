📘 Documentação do Sistema de Reconhecimento Facial
 Gustavo Felex RM554242
 Vinicius Santos RM552904
 Vinicius Issa Gois Rm553814
 Gustavo Bonani RM553493
 Wesley Leopoldino RM553496

🎯 Objetivo
Este projeto implementa um sistema completo de reconhecimento facial utilizando Python e OpenCV.
Ele permite:
Captura de imagens de usuários pela webcam
Treinamento de um modelo de reconhecimento facial (LBPH)
Identificação de pessoas em tempo real usando a webcam
Armazenamento de nomes, IDs e imagens para cada usuário cadastrado
O sistema foi projetado para ser simples, funcional e modular, permitindo capturar dados, treiná-los e realizar o reconhecimento facial de forma independente.
Execução / Como Usar
Ao rodar o programa, o usuário acessa um menu interativo, com as seguintes opções:
1 — Capturar faces
Solicita o ID e o nome do usuário.
Captura diversas imagens do rosto via webcam.
As imagens são salvas na pasta data/.
O nome e ID são registrados no arquivo names.json.
2 — Treinar modelo
Carrega todas as imagens da pasta data/.
Usa o algoritmo LBPH Face Recognizer do OpenCV para treinar o modelo.
O modelo final é salvo no arquivo model.yml.
3 — Reconhecer em tempo real
Carrega o modelo treinado.
Detecta rostos na webcam usando Haar Cascade.
Reconhece rostos conhecidos com base no modelo LBPH.
Exibe o nome e o nível de confiança na tela.
Rostos desconhecidos (confiança acima do limite) são marcados como “Desconhecido”.
0 — Sair
Fecha o programa.
📦 Dependências
O projeto utiliza as seguintes bibliotecas:
Biblioteca	Finalidade
OpenCV (cv2)	Detecção de rostos e reconhecimento LBPH
NumPy	Manipulação de arrays para o modelo
JSON	Armazenamento de nomes e IDs
OS	Organização de arquivos e pastas
Instalação recomendada:
pip install opencv-contrib-python
pip install numpy
Importante:
O LBPH só funciona com opencv-contrib-python (não funciona no pacote opencv normal).

Parâmetros do Sistema
O código possui vários parâmetros configuráveis no início:
Diretórios e arquivos
DATA_DIR = "data" → pasta onde ficam as imagens capturadas
MODEL_PATH = "model.yml" → arquivo do modelo treinado
NAMES_FILE = "names.json" → registro de nomes e IDs
Configuração da câmera
CAMERA_INDEX = 0 → 0 = webcam padrão
Parâmetros da detecção (Haar Cascade)
Parâmetro	Descrição
FACE_SCALE_FACTOR = 1.3	Redução progressiva para busca de rostos
FACE_MIN_NEIGHBORS = 5	Qualidade da detecção
FACE_MIN_SIZE = (100, 100)	Tamanho mínimo do rosto

Parâmetros do modelo LBPH
Parâmetro	Significado
LBPH_RADIUS = 1	Raio do LBPH
LBPH_NEIGHBORS = 8	Vizinhos na análise
LBPH_GRID_X = 8	Divisões horizontais da imagem
LBPH_GRID_Y = 8	Divisões verticais
Esses parâmetros influenciam diretamente na precisão do reconhecimento.
Reconhecimento
CONFIDENCE_THRESHOLD = 80.0
Valores abaixo = reconhecido
Valores acima = "Desconhecido"
