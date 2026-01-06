@echo off
echo Starting Wheat Disease Detection (Enhanced UI)...

cd /d D:\OneDrive\Desktop\Project\MAJOR_PROJECT\WheatDisease_MobileNetV2\WheatDisease_MobileNetV2
call wheat_env\Scripts\activate.bat
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

streamlit run app_wheat_enhanced.py

pause
