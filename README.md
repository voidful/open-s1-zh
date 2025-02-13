# open-s1-zh

```bash
conda create -n open-r1-zh python=3.10.9
conda activate open-r1-zh
conda install gxx_linux-64
conda install hcc::cudatoolkit
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
pip install -r requirements.txt
```