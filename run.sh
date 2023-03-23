conda create -n stgram_mfn python=3.7
conda activate stgram_mfn
# set tsinghua source
#pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# install package
pip install -r requirements.txt
# run
# train model
python run.py --load_epoch=False
# test best model
#python run.py --load_epoch=best
# test on evaluation dataset
cd evaluator
python evaluator.py
# visualization
#tensorboard --logdir=runs