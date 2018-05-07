# coding=utf-8

from __future__ import print_function

import random

import cPickle as pickle  ### https://docs.python.org/2/library/pickle.html

from utils import config


## general helper functions
class Data_Reader:
    def __init__(self, cur_train_index=0, load_list=False): ### check the load list thing
        self.training_data = pickle.load(open(config.training_data_path, 'rb')) ## load pickle
        self.data_size = len(self.training_data) ## check data

        ##make sure loaded list is kosher
        if load_list:
            self.shuffle_list = pickle.load(open(config.index_list_file, 'rb'))
        else:
            self.shuffle_list = self.shuffle_index() ## shuffle the index of the list
        self.train_index = cur_train_index ## curate the trained index



        ### check the batch number here
    def get_batch_num(self, batch_size):
        return self.data_size // batch_size

    #check the index out 00 use the random.sample function
    def shuffle_index(self):
        shuffle_index_list = random.sample(range(self.data_size), self.data_size) ## random.sample ####
        #https://stackoverflow.com/questions/22842289/generate-n-unique-random-numbers-within-a-range
        pickle.dump(shuffle_index_list, open(config.index_list_file, 'wb'), True) ## check the index list
        return shuffle_index_list

    ### get the right batch index

    def generate_batch_index(self, batch_size):
        if self.train_index + batch_size > self.data_size: ### train index
            batch_index = self.shuffle_list[self.train_index:self.data_size] ## check correct data size
            self.shuffle_list = self.shuffle_index() ## see shuffle list
            remain_size = batch_size - (self.data_size - self.train_index) ## remain size
            batch_index += self.shuffle_list[:remain_size]
            self.train_index = remain_size ### train index needs to increment
        else:
            batch_index = self.shuffle_list[self.train_index:self.train_index+batch_size]
            self.train_index += batch_size

        return batch_index

### get training batch correct
    ### can do different functions for each? Or, better to do unified.
    #####too hard all together.

    def generate_training_batch(self, batch_size):
        batch_index = self.generate_batch_index(batch_size) ###start
        batch_X = [self.training_data[i][0] for i in batch_index]   # batch_size of conv_a
        batch_Y = [self.training_data[i][1] for i in batch_index]   # batch_size of conv_b

        return batch_X, batch_Y


### training batch
    def generate_training_batch_with_former(self, batch_size):
        batch_index = self.generate_batch_index(batch_size)
        batch_X = [self.training_data[i][0] for i in batch_index]   # batch_size of conv_a
        batch_Y = [self.training_data[i][1] for i in batch_index]   # batch_size of conv_b
        former = [self.training_data[i][2] for i in batch_index]    # batch_size of former utterance

        return batch_X, batch_Y, former
### Testing batch
    def generate_testing_batch(self, batch_size):
        batch_index = self.generate_batch_index(batch_size)
        batch_X = [self.training_data[i][0] for i in batch_index]   # batch_size of conv_a

        return batch_X