python train_wavegan.py train .\train_swordfighter ^
--data_dir .\data\Sword_Fighter_16bit ^
--use_resnet ^
--data_first_slice ^
--data_fast_wav ^
--data_num_channels 2 ^
--wavegan_kernel_len 9 ^
--wavegan_disc_nupdates 1 ^
--train_batch_size 50 ^
--wavegan_dim 32 ^
--train_save_secs 600 ^
--train_summary_secs 15

REM --wavegan_disc_phaseshuffle 0 ^
REM --use_progressive_growing ^
REM --train_batch_size 32 ^
REM --wavegan_dim 52 ^
