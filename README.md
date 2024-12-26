# ENHAvir
 
<p align="center">
<img src="assets/ENHAvir pipeline.jpg" alt="ENHAvir Framework" width= "80%" class="center" >

The Overall framework of the ENHAvir. Enhancer sequences and activation values of the KSHV genome were used to train the DeBERTa Language Model of ENHAvir. Following that, ENHAvir can predict enhancers and their potential activation values of a given DNA sequence.


## Prerequisites

Our code is built on [pytorch 2.5.1+cu118](https://pytorch.org) and the [transformers](https://github.com/huggingface/transformers) library that you can install using `requirements.txt`.

```bash
git clone https://github.com/shamay-lab/ENHAvir
cd ENHAvir
pip install -r requirements.txt
```

Download `checkpoint.pt` file from [here](https://drive.google.com/file/d/18aCyrEsfTgwgnhNoXzIA_uJWv8Qjl5jZ/view?usp=sharing) and keep it in the `weights` directory.

## Using ENHAvir

You can use `predict.py` to perform predictions for a DNA sequence: 

```bash
CUDA_VISIBLE_DEVICES=0 python predict.py --filename="data/EBV.txt"
```

In this case, the DNA sequence is read from the `data/EBV.txt` file and the result is saved in the `results/EBV_predictions.csv` file.

Some more example commands are shown in `run.sh`. 
