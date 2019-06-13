python train_wavegan.py train .\train_gamemaster ^
--data_dir .\data\Gamemaster_Audio_16KHz ^
--use_resnet ^
--use_ortho_init ^
--use_conditioning ^
--use_skip_z ^
--use_spec_norm ^
--data_first_slice ^
--data_fast_wav ^
--data_num_channels 2 ^
--train_autobalance_classes ^
--wavegan_genr_upsample zeros ^
--wavegan_loss wgan-gp ^
--wavegan_disc_nupdates 1 ^
--wavegan_disc_phaseshuffle 1 ^
--wavegan_latent_dim 100 ^
--embedding_dim 100 ^
--wavegan_kernel_len 25 ^
--wavegan_dim 64 ^
--train_batch_size 64 ^
--n_minibatches 2 ^
--train_save_secs 600 ^
--train_summary_secs 15

REM --n_macro_patches 4 ^
REM --n_micro_patches 4 ^
