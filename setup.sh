## create uv env 

uv pip install -r requirements.txt
uv pip install "sglang[all]>=0.5.1.post2"
apt-get update && apt-get install -y python3.10-dev build-essential
apt-get install -y numactl