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
  <img src="screenshots/Screenshot_1.png" width="400" alt="Screenshot 1" />
  <img src="screenshots/Screenshot_2.png" width="400" alt="Screenshot 2" />
</p>
<p align="center>
  <img src="screenshots/Screenshot_3.png" width="400" alt="Screenshot 3" />
  <img src="screenshots/Screenshot_4.jpg" width="400" alt="Screenshot 4" />
</p>
<p align="center">
  <img src="screenshots/Screenshot_5.jpg" width="400" alt="Screenshot 5" />
  <img src="screenshots/Screenshot_6.jpg" width="400" alt="Screenshot 6" />
</p>
<p align="center>
  <img src="screenshots/Screenshot_7.jpg" width="400" alt="Screenshot 7" />
  <img src="screenshots/Screenshot_8.jpg" width="400" alt="Screenshot 8" />
</p>
<p align="center">
  <img src="screenshots/Screenshot_9.jpg" width="400" alt="Screenshot 9" />
  <img src="screenshots/Screenshot_10.jpg" width="400" alt="Screenshot 10" />
  <img src="screenshots/Screenshot_11.jpg" width="400" alt="Screenshot 11" />
</p>

---

🔄 Тестирование углов поворота

Проведено исследование влияния угла поворота штрих-кода относительно осей камеры:

При повороте на небольшой угол (до 15°) считывание поддерживается.

При повороте на угол > 15° считывание не происходит.

Чем ближе штрих-код к центральной оси X и Y кадра, тем выше вероятность успешного считывания.

Ниже приведены скриншоты для углов поворота:

<p align="center">
  <img src="screenshots/Screenshot_12.jpg" width="400" alt="Screenshot 9" />
  <img src="screenshots/Screenshot_13.jpg" width="400" alt="Screenshot 10" />
</p>

## 📑 Зависимости

```text
pyzbar
opencv-python
numpy
```

---
