import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm

def get_datasets(dataset_name='MNIST'):
    if dataset_name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        # Per MNIST, il transform di calibrazione è lo stesso
        calib_transform = transform

    elif dataset_name == 'CIFAR10':
        # Trasformazioni specifiche per CIFAR-10
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

        # Per la calibrazione, usiamo le trasformazioni di test (senza augmentation casuale)
        calib_transform = transform_test

    else:
        raise ValueError(f"Dataset '{dataset_name}' not supported")

    # Creiamo un piccolo subset per la calibrazione dell'allineamento
    calib_indices = np.random.choice(len(train_dataset), 512, replace=False)

    # Crea il dataset di calibrazione usando il transform corretto
    if dataset_name == 'MNIST':
         calib_dataset = Subset(train_dataset, calib_indices)
    else: # CIFAR10
         # Dobbiamo ricaricare il training set senza augmentation per la calibrazione
         train_dataset_for_calib = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=calib_transform)
         calib_dataset = Subset(train_dataset_for_calib, calib_indices)

    return train_dataset, test_dataset, calib_dataset

def train_model(model, train_loader, device, epochs=5, weight_decay=5e-4):
    """
    Funzione per addestrare un modello (versione migliorata con scheduler e weight decay).
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for i, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                pbar.set_postfix({'loss': f'{running_loss / 100:.3f}', 'lr': f'{scheduler.get_last_lr()[0]:.5f}'})
                running_loss = 0.0

        scheduler.step()

    print('Finished Training')
    return model

def evaluate_model(model, test_loader, device):
    """
    Funzione per valutare un modello, restituisce loss e accuratezza.
    """
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(test_loader)
    return avg_loss, accuracy

def evaluate_interpolation(model_arch, state_dict_a, state_dict_b, test_loader, device, n_points=21):
    """
    Esegue l'interpolazione lineare (LERP) tra due state_dict e valuta ogni punto.
    """
    alphas = torch.linspace(0, 1, n_points)
    losses = []
    accuracies = []

    pbar = tqdm(alphas, desc="Evaluating LERP Interpolation")
    for alpha in pbar:
        # Crea lo state_dict interpolato facendo la media ponderata
        interpolated_state_dict = {}
        for key in state_dict_a.keys():
            interpolated_state_dict[key] = (1 - alpha) * state_dict_a[key] + alpha * state_dict_b[key]

        # Crea un modello temporaneo e carica i pesi interpolati
        temp_model = model_arch().to(device)
        temp_model.load_state_dict(interpolated_state_dict)

        # Valuta il modello interpolato
        loss, acc = evaluate_model(temp_model, test_loader, device)
        losses.append(loss)
        accuracies.append(acc)
        pbar.set_postfix({'alpha': f'{alpha:.2f}', 'loss': f'{loss:.3f}', 'acc': f'{acc:.2f}%'})

    return alphas.numpy(), np.array(losses), np.array(accuracies)

def compare_predictions(model1, model2, data_loader, device):
    """Calcola la percentuale di accordo tra le predizioni di due modelli."""
    model1.to(device)
    model2.to(device)
    model1.eval()
    model2.eval()
    matches = 0
    total = 0
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            pred1 = torch.argmax(model1(images), dim=1)
            pred2 = torch.argmax(model2(images), dim=1)
            matches += (pred1 == pred2).sum().item()
            total += images.size(0)
    return 100 * matches / total

def calculate_logit_distance(model1, model2, data_loader, device):
    """Calcola l'errore quadratico medio (MSE) tra i logits di due modelli."""
    model1.to(device)
    model2.to(device)
    model1.eval()
    model2.eval()
    total_mse = 0.0
    num_batches = 0
    with torch.no_grad():
        for images, _ in data_loader:
            images = images.to(device)
            logits1 = model1(images)
            logits2 = model2(images)
            mse = nn.functional.mse_loss(logits1, logits2)
            total_mse += mse.item()
            num_batches += 1
    return total_mse / num_batches

def slerp(p0_flat, p1_flat, t, eps=1e-8):
    """
    Interpolazione Lineare Sferica tra due tensori 1D (flattened).
    """
    # I tensori di input sono già 1D e sullo stesso dispositivo
    # Normalizza i vettori per proiettarli sull'ipersfera
    p0_norm = p0_flat / (torch.norm(p0_flat) + eps)
    p1_norm = p1_flat / (torch.norm(p1_flat) + eps)

    # Calcola il coseno dell'angolo tra di loro
    omega = torch.acos(torch.dot(p0_norm, p1_norm).clamp(-1, 1))

    if torch.abs(omega) < eps:
        # Se sono molto vicini, ritorna una semplice interpolazione lineare
        return (1.0 - t) * p0_flat + t * p1_flat

    so = torch.sin(omega)
    # Formula SLERP
    interp_flat = (torch.sin((1.0 - t) * omega) / so) * p0_norm + (torch.sin(t * omega) / so) * p1_norm

    # Ritorna il tensore interpolato 1D. Il chiamante lo rimetterà in forma.
    return interp_flat

def slerp_evaluate_interpolation(model_arch, state_dict_a, state_dict_b, test_loader, device, n_points=21):
    alphas = torch.linspace(0, 1, n_points)
    losses = []
    accuracies = []

    pbar = tqdm(alphas, desc="Evaluating SLERP Interpolation")
    for alpha in pbar:
        interpolated_state_dict = {}
        for key in state_dict_a.keys():
            p_a = state_dict_a[key]
            p_b = state_dict_b[key]
            original_shape = p_a.shape

            # Appiattisci, interpola, e poi rimetti in forma
            p_a_flat = p_a.flatten().to(device)
            p_b_flat = p_b.flatten().to(device)

            interp_flat = slerp(p_a_flat, p_b_flat, alpha.item())

            # Mantieni la magnitudine originale interpolando linearmente le norme
            norm_a = torch.norm(p_a_flat)
            norm_b = torch.norm(p_b_flat)
            interp_norm = (1.0 - alpha.item()) * norm_a + alpha.item() * norm_b

            # Applica la norma interpolata
            interp_param = interp_flat * (interp_norm / (torch.norm(interp_flat) + 1e-8))

            interpolated_state_dict[key] = interp_param.reshape(original_shape)

        temp_model = model_arch().to(device)
        temp_model.load_state_dict(interpolated_state_dict)

        loss, acc = evaluate_model(temp_model, test_loader, device)
        losses.append(loss)
        accuracies.append(acc)
        pbar.set_postfix({'alpha': f'{alpha:.2f}', 'loss': f'{loss:.3f}', 'acc': f'{acc:.2f}%'})

    return alphas.numpy(), np.array(losses), np.array(accuracies)