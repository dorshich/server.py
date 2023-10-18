cd C:\EquashieldVialLearning\Backend\venv
powershell -Command "(gc pyvenv.cfg) -replace 'UserName', '%USERNAME%' | Out-File -encoding ASCII pyvenv.cfg"
cd C:\EquashieldVialLearning\Backend\scripts_models
"C:\EquashieldVialLearning\Backend\scripts_models\venv\Scripts\python.exe" server.py
