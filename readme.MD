
python -m venv myenv
source myenv/bin/activate

pip install -r requirements.txt
pip install "fastapi[standard]"

python -m fastapi dev app.py

http://localhost:8000/docs#/default/query_rag_query_post