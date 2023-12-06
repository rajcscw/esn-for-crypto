Code for the paper:

https://arxiv.org/pdf/1704.01046.pdf


# Create new environment
```
conda create -n esn-crypto python=3.7
conda activate esn-crypto
```
# Install dependencies

Go into the folder and install the dependencies

```
cd esn-for-crypto
pip install -e .
```

# Simulate the encryption and decryption
Change directory to the scripts/encrypt-decrypt folder and run `simulate.py`

```
cd scripts/encrypt-decrypt
python simulate.py
```

