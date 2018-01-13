# attentional_factorization_machine

This is our implementation for the paper:

Jun Xiao, Hao Ye, Xiangnan He, Hanwang Zhang, Fei Wu and Tat-Seng Chua (2017). [Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks](http://www.comp.nus.edu.sg/~xiangnan/papers/ijcai17-afm.pdf) IJCAI, Melbourne, Australia, August 19-25, 2017.

We have additionally released our TensorFlow implementation of Factorization Machines under our proposed neural network framework. 

**Please cite our IJCAI'17 paper if you use our codes. Thanks!** 

Author: Xiangnan He (xiangnanhe@gmail.com) and Hao Ye (haoyev@gmail.com)

## Environments
* Tensorflow (version: 1.0.1)
* numpy
* sklearn
## Dataset
We use the same input format as the LibFM toolkit (http://www.libfm.org/). In this instruction, we use [MovieLens](grouplens.org/datasets/movielens/latest).
The MovieLens data has been used for personalized tag recommendation, which contains 668,953 tag applications of users on movies. We convert each tag application (user ID, movie ID and tag) to a feature vector using one-hot encoding and obtain 90,445 binary features. The following examples are based on this dataset and it will be referred as ***ml-tag*** wherever in the files' name or inside the code.
When the dataset is ready, the current directory should be like this:
* code
    - AFM.py
    - FM.py
    - LoadData.py
* data
    - ml-tag
        - ml-tag.train.libfm
        - ml-tag.validation.libfm
        - ml-tag.test.libfm

## Quick Example with Optimal parameters
Use the following command to train the model with the optimal parameters:
```
# step into the code folder
cd code
# train FM model and save as pretrain file
python FM.py --dataset ml-tag --epoch 100 --pretrain -1 --batch_size 4096 --hidden_factor 256 --lr 0.01 --keep 0.7
# train AFM model using the pretrained weights from FM
python AFM.py --dataset ml-tag --epoch 100 --pretrain 1 --batch_size 4096 --hidden_factor [8,256] --keep [1.0,0.5] --lamda_attention 2.0 --lr 0.1
```
The instruction of commands has been clearly stated in the codes (see the parse_args function). 

The current implementation supports regression classification, which optimizes RMSE. 

## Performance Comparison
### Parameters
For the sake of a quick demonstration for the improvement of our AFM model compared to original FM, we set the dimension of the ***embedding factor*** to be 16 (instead of 256 in our paper), and ***epoch*** as 20. 

### Train
Step into the ***code*** folder and train FM and AFM as follows. This will start to train our AFM model on the dataset ***frappe*** based on the pretrained model of FM. The parameters have been initialized optimally according to our experiments. It will loop 20 epochs and print the best epoch depending on the validation result.
```
# step into the code folder
cd code
# train FM model with optimal parameters
python FM.py --dataset ml-tag --epoch 20 --pretrain -1 --batch_size 4096 --hidden_factor 16 --lr 0.01 --keep 0.7
# train AFM model with optimal parameters
python AFM.py --dataset ml-tag --epoch 20 --pretrain 1 --batch_size 4096 --hidden_factor [16,16] --keep [1.0,0.5] --lamda_attention 100.0 --lr 0.1
```
After the trainning processes finish, the trained models will be saved into the ***pretrain*** folder, which should be like this:
* pretrain
    - afm_ml-tag_16
        - checkpoint
        - ml-tag_16.data-00000-of-00001
        - ml-tag_16.index
        - ml-tag_16.meta
    - fm_ml-tag_16
        - checkpoint
        - ml-tag_16.data-00000-of-00001
        - ml-tag_16.index
        - ml-tag_16.meta
### Evaluate
Now it's time to evaluate the pretrained models with the test datasets, which can be done by running ***AFM.py*** and ***FM.py*** with ***--process evaluate*** as follows:
```
# evaluate the pretrained FM model
python FM.py --dataset ml-tag --epoch 20 --batch_size 4096 --lr 0.01 --keep 0.7 --process evaluate
# evaluate the pretrained AFM model
python AFM.py --dataset ml-tag --epoch 20 --pretrain 1 --batch_size 4096 --hidden_factor [16,16] --keep [1.0,0.5] --lamda_attention 100.0 --lr 0.1 --process evaluate
```

Last Update Date: Aug 2, 2017
