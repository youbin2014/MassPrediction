# MassPrediction

To use run the code:

1. Install Anaconda
2. `conda create -n MP python=3.8`
3. `conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`
4. `python MoumouPredictionBNN.py --test-batch-size=1
    --mode="train"
    --save_dir="./"
    --num_monte_carlo=50
    --lr=0.1
    --epochs=1000`
