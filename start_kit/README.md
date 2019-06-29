# Fashion-IQ Starter Code
## About this repository
For more information on the dataset, please visit its [project page](https://www.spacewu.com/posts/fashion-iq). 
## Starter code for Fashion IQ challenge 
To get started with the framework, install the following dependencies:
- Python 3.6
- [PyTorch 0.4](https://pytorch.org/get-started/previous-versions/)
## Train and evaluate a model
Follow the following steps to train a model:
1. Download the dataset and resize the images. 
2. Build the vocabulary for a specific datasest:
```
python build_vocab.py --data_set dress
```
3. Train the model 
```
python train.py --data_set dress --batch_size 128 --log_step 15
```
The trained models will be saved into the folder `models/` every epoch. 
4. Generate the submission results 
```
python eval.py --data_set dress --batch_size 128 --model_folder <your_model_folder_name> --data_split test
```
5. Submit your results at: https://competitions.codalab.org/competitions/23391#results
