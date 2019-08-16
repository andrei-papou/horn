cd python_cli || echo 'No "python_cli" directory!'
pip install --quiet -r requirements.txt
python main.py "$@"
cd ..
