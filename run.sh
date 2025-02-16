# Results reported in the paper with default setting:
CUDA_VISIBLE_DEVICES=0 python predict.py --filename="data/SV40.txt"

# Perform prediction with a sliding length of 100: 
CUDA_VISIBLE_DEVICES=0 python predict.py --slide 100 --filename="data/hg38_Fos.txt" --outfilename="hg38_Fos_predictions_slide_100.csv"

# Perform post-processing with a threshold percentile of 75: 
CUDA_VISIBLE_DEVICES=0 python predict.py --threshold 75 --filename="data/hg38_Jun.txt" --outfilename="hg38_Jun_predictions_percentile_75.csv"