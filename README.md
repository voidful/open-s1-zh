#Open-r1-zh

```bash
conda create -n open-r1-zh python=3.10.9
conda activate open-r1-zh
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
conda install cuda-nvcc -c nvidia
pip install -r requirements.txt
```