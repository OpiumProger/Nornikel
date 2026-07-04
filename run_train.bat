@echo off
REM Обучение U-Net в conda new_chemberta_env
cd /d "%~dp0"

call C:\Users\USER\anaconda3\Scripts\activate.bat new_chemberta_env

echo === Проверка окружения ===
python scripts\check_env.py
echo.

echo === Установка недостающих пакетов (если нужно) ===
pip install segmentation-models-pytorch albumentations opencv-python-headless --quiet

echo.
echo === Обучение ===
python train.py --epochs 25 --batch-size 2 --patch-size 384

pause
