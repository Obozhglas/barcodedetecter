# Importing library
import logging

import cv2
from pyzbar.pyzbar import decode
import numpy as np

from settings import CAMERA

logging.basicConfig(
    level=logging.INFO,
    format='[%(filename)s:%(lineno)d] [%(module)s] %(levelname)s %(funcName)s : %(message)s'
)

def camera_capture():
    # Инициализация камеры (0 - обычно встроенная или первая подключённая камера)
    cap = cv2.VideoCapture(CAMERA)
    not_found_logged = False

    if not cap.isOpened():
        print("Не удалось открыть камеру")
        return

    while True:
        # Считываем кадр с камеры
        ret, frame = cap.read()

        if not ret:
            print("Не удалось получить кадр. Выход...")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Повышение контрастности
        frame = cv2.equalizeHist(frame)

        detectedBarcodes = decode(frame)

        # Если штрихкоды найдены, обрабатываем каждый
        if detectedBarcodes:
            not_found_logged = False
            for barcode in detectedBarcodes:

                corners = barcode.polygon

                if len(corners) == 4:
                    corners_array = np.array([(p.x, p.y) for p in corners], dtype=np.int32)

                    # Рисуем ломаную по реальным углам (точная граница)
                    cv2.polylines(
                        frame,
                        [corners_array],  # формат: список контуров
                        isClosed=True,
                        color=(0, 255, 0),
                        thickness=2
                    )

                    # Получаем прямоугольник, охватывающий все эти точки
                    x, y, w, h = cv2.boundingRect(corners_array)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                if barcode.data:
                    logging.info(f"Найден штрихкод: {barcode.data.decode('utf-8')}")
                    logging.info(f"Тип: {barcode.type}")
        else:
            if not not_found_logged:
                logging.info("Штрихкод не был найден.")
                not_found_logged = True

        # Отображаем кадр в окне
        cv2.imshow("Barcode Scanner", frame)

        # Выход из цикла при нажатии 'q' или ESC
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    # Освобождаем ресурс камеры и закрываем окна
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    camera_capture()
