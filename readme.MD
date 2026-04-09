python -m venv myenv
source myenv/bin/activate

pip install "fastapi[standard]"

python -m fastapi dev app.py

fastapi dev app.py

conda deactivate
source myenv/bin/activate
python -m pip install -r requirements.txt
python -m pip install "fastapi[standard]"
python -m fastapi dev app.py

http://localhost:8000/docs#/default/query_rag_query_post