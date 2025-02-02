import torch
from safetensors.torch import load_file
from torchvision import models, transforms
from PIL import Image

# Пути
MODEL_PATH = "../model/resnet50-smokedataset/model.safetensors"
# IMAGE_PATH = "image/fire.jpg"

# Создаем модель ResNet50
model = models.resnet50(pretrained=False)

# Изменяем последний слой для 2 классов
num_classes = 2  # Укажите количество классов в вашем датасете
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Загружаем веса
state_dict = load_file(MODEL_PATH)

# Удаляем ключи, связанные с последним слоем, чтобы избежать ошибки размера
state_dict.pop("fc.weight", None)
state_dict.pop("fc.bias", None)

# Загружаем оставшиеся веса
model.load_state_dict(state_dict, strict=False)  # Игнорируем несоответствующие ключи

# Перемещение модели на GPU (если доступно)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()  # Переводим модель в режим оценки

# Преобразования для входного изображения
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet50 требует изображения 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Нормализация для ImageNet
])

# Функция для загрузки и обработки изображения
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')  # Конвертируем в RGB
    image = transform(image).unsqueeze(0)  # Добавляем batch-размерность
    return image.to(device)  # Перемещаем изображение на GPU (если доступно)

# Функция для предсказания
def predict(image_path):
    # Подготовка изображения
    image = preprocess_image(image_path)

    # Предсказание
    with torch.no_grad():  # Отключаем вычисление градиентов
        output = model(image)
        _, predicted = torch.max(output, 1)  # Получаем предсказанный класс

    return predicted.item()

# Основная функция
# def main():
#     # Предсказание на тестовом изображении
#     predicted_class = predict(IMAGE_PATH)
#     print(f"Предсказанный класс: {predicted_class}")
