python train_wavegan.py train .\train_swordfighter ^
--data_dir .\data\Sword_Fighter_16bit ^
--data_first_slice ^
--data_fast_wav ^
--data_num_channels 2 ^
--wavegan_dim 52 ^
--train_batch_size 32 ^
--use_progressive_growing ^
--train_save_secs 600 ^
--train_summary_secs 15

REM --use_progressive_growing ^