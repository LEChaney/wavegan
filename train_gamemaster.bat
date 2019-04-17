python train_wavegan.py train .\train_gamemaster ^
--data_dir .\data\Gamemaster_Audio_16KHz ^
--data_first_slice ^
--data_fast_wav ^
--data_num_channels 2 ^
--wavegan_dim 32 ^
--data_slice_len 65536 ^
--train_save_secs 600 ^
--train_summary_secs 15