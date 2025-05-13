# Машинное зрение для распознавания штрих-кодов

Этот проект захватывает видеопоток с веб‑камеры, находит штрих-коды на кадрах и выводит информацию о распознанном коде и его типе.

---

## 📂 Структура проекта

```
├── main.py            # Основной скрипт для захвата и обработки видео
├── requirements.txt   # Зависимости проекта
├── settings.py        # Настраивается источник видеопотока
└── screenshots/       # Папка с примерами работы (11 изображений)
    ├── screenshot1.jpg
    ├── screenshot2.jpg
    ├── screenshot3.jpg
    ├── screenshot4.jpg
    ├── screenshot5.jpg
    ├── screenshot6.jpg
    ├── screenshot7.jpg
    ├── screenshot8.jpg
    ├── screenshot9.jpg
    ├── screenshot10.jpg
    └── screenshot11.jpg
```

## 🛠 Установка

1. Клонируйте репозиторий:

   ```bash
   git clone <URL вашего репо>
   cd <папка проекта>
   ```
2. Создайте виртуальное окружение и установите зависимости:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\\Scripts\\activate   # Windows
   pip install -r requirements.txt
   ```

> **Требования:** Python 3.7+, OpenCV, PyZbar, NumPy

## 🚀 Запуск

```bash
python main.py
```

* Откроется окно с видеопотоком с веб‑камеры (1080p, 30 FPS) (в моем случае).
* В терминале появляются лог‑сообщения о найденных штрих‑кодах и их типах.
* Для остановки нажмите `Ctrl+C` или закройте окно.

## 📸 Примеры работы

Ниже приведены 11 скриншотов, иллюстрирующих работу алгоритма в разных условиях (освещение, фон, расстояние).

<p align="center">
  <img src="screenshots/Screenshot1.png" width="200" alt="Screenshot 1" />
  <img src="screenshots/Screenshot2.png" width="200" alt="Screenshot 2" />
  <img src="screenshots/Screenshot3.png" width="200" alt="Screenshot 3" />
  <img src="screenshots/Screenshot4.jpg" width="200" alt="Screenshot 4" />
</p>
<p align="center">
  <img src="screenshots/Screenshot5.jpg" width="200" alt="Screenshot 5" />
  <img src="screenshots/Screenshot6.jpg" width="200" alt="Screenshot 6" />
  <img src="screenshots/Screenshot7.jpg" width="200" alt="Screenshot 7" />
  <img src="screenshots/Screenshot8.png" width="200" alt="Screenshot 8" />
</p>
<p align="center">
  <img src="screenshots/Screenshot9.jpg" width="200" alt="Screenshot 9" />
  <img src="screenshots/Screenshot10.jpg" width="200" alt="Screenshot 10" />
  <img src="screenshots/Screenshot11.jpg" width="200" alt="Screenshot 11" />
</p>

---

## 📑 Зависимости

```text
pyzbar
opencv-python
numpy
```

---
