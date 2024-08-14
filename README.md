# AOW-Autoregressive Out-of-distribution Watermarking

Implementation for [Watermarking Recommender Systems](https://arxiv.org/pdf/2407.21034), CIKM24.


## Requirements
Python==3.7.16 and PyTorch==1.13.1.


## Train Oracle Models

```bash
python train.py --device cuda:0 --dataset_code ml-1m --gold --model_code bert
```

## Train & Test Watermarked Models

```bash
python train.py --device cuda:0 --dataset_code ml-1m --model_code bert --number_ood_seqs 0.1 --number_ood_val_seqs 1.0 --pattern_len 5 --bottom_m 100 --method cold
python test_watermark_acc.py --device cuda:0 --dataset_code ml-1m --model_code bert --number_ood_seqs 0.1 --number_ood_val_seqs 1.0 --pattern_len 5 --bottom_m 100 --method cold
```
Note: you can evaluate the watermark on the oracle model by changing the parameters of test_watermark_acc.py. This also works for the following distllation and fine-tuning processes.

## Distill & Test the Watermarked Model

```bash
python distill.py --device cuda:0 --dataset_code ml-1m --model_code bert --bb_model_code bert --number_ood_seqs 0.1 --number_ood_val_seqs 1.0 --pattern_len 5 --bottom_m 100 --method cold
python test_watermark_acc_distilled.py --device cuda:0 --dataset_code ml-1m --model_code bert --bb_model_code bert --number_ood_seqs 0.1 --number_ood_val_seqs 1.0 --pattern_len 5 --bottom_m 100 --method cold
```

## Fine-tune & Test the Watermarked Model

```bash
python finetune.py --device cuda:0 --dataset_code ml-1m --model_code bert --number_ood_seqs 0.1 --number_ood_val_seqs 1.0 --pattern_len 5 --bottom_m 100 --method cold --finetune_ratio 0.1
python test_watermark_acc_afterfinetune.py --device cuda:0 --dataset_code ml-1m --model_code bert --number_ood_seqs 0.1 --number_ood_val_seqs 1.0 --pattern_len 5 --bottom_m 100 --method cold --finetune_ratio 0.1
```

## Citation
Please cite the following paper if you use our methods in your research:
```
@article{zhang2024watermarking,
  title={Watermarking Recommender Systems},
  author={Zhang, Sixiao and Long, Cheng and Yuan, Wei and Chen, Hongxu and Yin, Hongzhi},
  journal={arXiv preprint arXiv:2407.21034},
  year={2024}
}
```

### Acknowledgement
Thanks to Yue for their great [work](https://github.com/Yueeeeeeee/RecSys-Extraction-Attack)! 