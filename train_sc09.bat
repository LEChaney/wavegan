python train_wavegan.py train .\train_sc09 ^
--data_dir .\data\sc09\train ^
--use_resnet ^
--use_ortho_init ^
--use_conditioning ^
--use_skip_z ^
--use_specnorm ^
--wavegan_loss hinge ^
--wavegan_batchnorm ^
--data_first_slice ^
--data_fast_wav ^
--wavegan_disc_nupdates 1 ^
--wavegan_disc_phaseshuffle 1 ^
--wavegan_latent_dim 100 ^
--embedding_dim 100 ^
--wavegan_kernel_len 25 ^
--wavegan_dim 64 ^
--train_batch_size 64 ^
--n_minibatches 1 ^
--train_save_secs 600 ^
--train_summary_secs 15
