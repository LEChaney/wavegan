@Echo off
:Start
set CUDA_VISIBLE_DEVICES="-1"
python train_wavegan.py incept ./train_sc09 --use_conditioning
echo Program terminated at %Date% %Time% with Error %ErrorLevel%
echo Press Ctrl-C if you don't want to restart automatically
ping -n 10 localhost

goto Start
