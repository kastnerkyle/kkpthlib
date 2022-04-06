 python sample_transformer_xl_raster_music_model.py /home/kkastner/_kkpthlib_models/transformer_xl_raster_music_model_07-13-57_2020-06-05_62a624_no_eos/saved_models/permanent_model-310944.pth -s 1024 -c 5


timidifyit.sh

file=$1
tempo=${2:-160}
timidity -T $tempo --output-24bit -Ow $file
bn=${file%.midi}
ffmpeg -i $bn.wav -acodec pcm_s16le -ar 44100 $bn_1.wav
mv $bn_1.wav $bn.wav
ffmpeg -i $bn.wav $bn.mp3
