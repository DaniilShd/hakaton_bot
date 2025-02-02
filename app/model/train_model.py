# Установка необходимых библиотек
# !pip install transformers datasets torch Pillow

# Импорт библиотек
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from datasets import load_dataset
from transformers import ResNetForImageClassification
from PIL import Image

# Загрузка датасета
dataset = load_dataset("sagecontinuum/smokedataset")

# Определение преобразований для изображений
transform = Compose([
    Resize((224, 224)),  # ResNet50 принимает изображения 224x224
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Нормализация для ImageNet
])

# Функция для применения преобразований
def preprocess(example):
    example["image"] = [transform(image.convert("RGB")) for image in example["image"]]
    return example

# Применяем преобразования к датасету
dataset = dataset.map(preprocess, batched=True)

# Преобразуем датасет в формат PyTorch
dataset.set_format(type="torch", columns=["image", "label"])

# Создаем DataLoader
train_dataloader = DataLoader(dataset["train"], batch_size=32, shuffle=True)
eval_dataloader = DataLoader(dataset["validation"], batch_size=32)

# Загрузка предобученной модели ResNet50
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

# Изменение классификатора под количество классов в датасете
num_classes = len(dataset["train"].features["label"].names)

# Получаем количество входных признаков для классификатора
# in_features = model.classifier.in_features  # Получаем размер входного слоя классификатора

# Получаем количество входных признаков для классификатора
if hasattr(model.classifier, "in_features"):
    in_features = model.classifier.in_features
else:
    # Если классификатор обернут в Sequential, находим Linear слой внутри
    for layer in model.classifier:
        if isinstance(layer, torch.nn.Linear):
            in_features = layer.in_features
            break

print(in_features)

# Заменяем классификатор
model.fc = torch.nn.Linear(in_features, num_classes)

# Оптимизатор и функция потерь
optimizer = Adam(model.parameters(), lr=1e-4)
criterion = CrossEntropyLoss()

# Перемещение модели на устройство (GPU или CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Обучение модели
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        inputs = batch["image"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()

        # Исправляем форму данных
        # inputs = inputs.view(-1, 2048)  # Теперь размерность (32, 2048)

        outputs = model(inputs).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# Оценка модели
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch in eval_dataloader:
        inputs = batch["image"].to(device)
        labels = batch["label"].to(device)

        outputs = model(inputs).logits
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Сохранение модели
model.save_pretrained("resnet50-smokedataset")