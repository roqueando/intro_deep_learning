from arch import AlgebricDataset, LineNetwork
from torch.nn import MSELoss
from torch.utils.data import DataLoader
import torch

device = "cpu"  # Configurando para treinar a rede no M1


def line_function(x): return 2*x + 3


# Definindo o tamanho do conjunto de amostras
train_samples = 1000
test_samples = 100

# Gerando os datasets (conjunto de dados) de treino e teste
# Com intervalo de -10 a 10, passando a quantidade de amostras
train_set = AlgebricDataset(line_function, (-10, 10), train_samples)
test_set = AlgebricDataset(line_function, (-10, 10), test_samples)

# Carregando o dataset e embaralhando o mesmo
# O tamanho do lote que será treinado por ser pequeno pode ser todo o conjunto
# Mas em outras arquiteturas ou como CNN que carrega imagens
# Normalmente são muitos como 70 mil ou mais imagens, logo o tamanho do lote
# tem de ser menor para que ele treine lote por lote sem esgotar os recursos do CPU
train_loader = DataLoader(train_set, batch_size=train_samples, shuffle=True)
test_loader = DataLoader(test_set, batch_size=test_samples, shuffle=True)

# Instanciando o modelo de arquitetura da Rede
model = LineNetwork().to(device)
loss_function = MSELoss()


# Função de treinamento da rede neural
def train(model: LineNetwork, dataloader: DataLoader, loss_function):
    model.train()
    cumloss = 0.0
    optimizer = model.optimizer(1e-3)
    for inputs, labels in dataloader:
        data_input = inputs.unsqueeze(1).float().to(device)
        data_label = labels.unsqueeze(1).float().to(device)

        output = model(data_input)
        loss = loss_function(output, data_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cumloss += loss.item()
    return cumloss / len(dataloader)


def test(model: LineNetwork, dataloader: DataLoader, loss_function):
    model.eval()

    cumloss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            data_input = inputs.unsqueeze(1).float().to(device)
            data_label = labels.unsqueeze(1).float().to(device)

            output = model(data_input)
            loss = loss_function(output, data_label)
            cumloss += loss.item()
    return cumloss / len(dataloader)
