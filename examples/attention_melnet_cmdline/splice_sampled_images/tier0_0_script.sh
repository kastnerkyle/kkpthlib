CUDA_VISIBLE_DEVICES=0 python sample_attention_melnet_cmdline.py  --axis_split=21212 --tier_input_tag=0,0 --size_at_depth=88,32 --n_layers=5 --hidden_size=256 --cell_type=gru --learning_rate=2E-5 --optimizer=adam --real_batch_size=1 --virtual_batch_size=1 --use_longest --experiment_name=attn_tts_robovoice_paper_mae_fp16_adam_gru_256_ramplr_round27 /home/kkastner/_kkpthlib_models/attention_melnet_cmdline_22-41-38_2021-24-07_07797f_attn_tts_robovoice_paper_mae_fp16_adam_gru_256_ramplr_round27_tier_0_0_sz_88_32/saved_models/valid_model-85332.pth

