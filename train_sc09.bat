python train_wavegan.py train .\train_sc09 ^
--data_dir .\data\sc09\train ^
--use_resnet ^
--use_conditioning ^
--data_first_slice ^
--data_fast_wav ^
--wavegan_disc_nupdates 1 ^
--wavegan_disc_phaseshuffle 1 ^
--wavegan_dim 64 ^
--train_save_secs 600 ^
--train_summary_secs 15
