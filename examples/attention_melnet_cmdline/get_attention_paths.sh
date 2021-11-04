echo "Filepath $1"
echo "Seq args $2 $3 $4"
i=$2
j=$3
k=$4
m=`seq $i $j $k`

#for l in $m; do
#    rm -rf plot_attention_paths/attn_$l
#done

rm plot_attention_paths/attn_*.png
rm plot_attention_paths/attn_path_*.gif

for l in $m; do
    python sample_attention_melnet_cmdline.py --direct_saved_model_path=$1/saved_models/checkpoint_model-$l.pth --axis_split=21212 --tier_input_tag=0,0 --size_at_depth=88,32 --n_layers=5 --hidden_size=256 --cell_type=gru --learning_rate=2E-5 --optimizer=adam --real_batch_size=16 --virtual_batch_size=16 --output_dir=plot_attention_paths --experiment_name=attn_tts_robovoice_paper_impl --terminate_early_attention_plot
    mkdir -p plot_attention_paths/attn_$l
    mv plot_attention_paths/attn_*.png plot_attention_paths/attn_$l/
done

attn_paths=()
for l in $m; do
    attn_paths+=(plot_attention_paths/attn_$l/attn_0.png)
done

echo ${attn_paths[@]}
convert -resize 50% -delay 100 -loop 0 `echo ${attn_paths[@]}` plot_attention_paths/attn_path_0.gif
