# Emotional Speech and Chat 

This project aims to implement and train all of the component models necessary for a chatbot which focuses on emotional relevance in chat. 

The dialogue generation is completed using a reinforcement learning model. 

The emotion recognition model classifies an input audio signal into one of six emotions: 
* happy
* sad 
* frustrated 
* neutral 
* angry 
* excited 

The speech recognition model is a reimplementation of Listen, Attend, and Spell by William Chen, et. al 

Training functions are located in Jupyter notebooks for ease of running. 

#### Datasets used for training 

* IEMOCAP emotional dialogue: emotional speech recognition 
* Mozilla Common Voice: speech recognition 
* Cornell Movie Dialog Corpus: conversation generation

#### Directories

* rl_model: reinforcement learning models and training 
* speech_models: emotion and speech recognition models 
* utils: feature extraction and data loading helpers 
    
#### Key References 

[1] Chan, William, et al. "Listen, attend and spell." (2015). 

[2] Chernykh, Vladimir, Grigoriy Sterling, and Pavel Prihodko. 
"Emotion recognition from speech with recurrent neural networks." 
arXiv preprint arXiv:1701.08071 (2017).

[3] Li, Jiwei, et al. "Deep reinforcement learning for dialogue generation." 
rXiv preprint arXiv:1606.01541 (2016).




