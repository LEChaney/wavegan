python train_wavegan.py train .\train_gamemaster ^
--data_dir .\data\Gamemaster_Audio_16KHz ^
--use_resnet ^
--use_conditioning ^
--data_first_slice ^
--data_fast_wav ^
--data_num_channels 2 ^
--wavegan_disc_nupdates 1 ^
--wavegan_disc_phaseshuffle 1 ^
--wavegan_dim 64 ^
--train_batch_size 110 ^
--train_save_secs 600 ^
--train_summary_secs 15
