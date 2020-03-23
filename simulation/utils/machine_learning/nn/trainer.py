
import database.dataset as ds

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from nn.train_callbacks import TensorboardVisualizerCallback, TensorboardLoggerCallback, ModelSaverCallback
from nn.test_callbacks import PredictionsSaverCallback
import helpers

from multiprocessing import cpu_count

import os

class Trainer:
    """
        A dataset loader taking a segmentation images as argument and return
        as them as tensors from getitem()

        Parameters
        ----------
        segmentation_images: list of SegmentationImage
            images contained in the dataset

        img_resize: Size which images will be resized to
    """
    def __init__(self, network, classifier):
        self.net = network
        self.classifier = classifier
    def get_callbacks(self):

        data_folder = os.path.join(os.environ.get('KITCAR_REPO_PATH'), 'kitcar-simulation-data')
        tb_viz_cb = TensorboardVisualizerCallback(os.path.join(data_folder, 'logs/tb_viz'))
        tb_logs_cb = TensorboardLoggerCallback(os.path.join(data_folder, 'logs/tb_logs'))
        model_saver_cb = ModelSaverCallback(os.path.join(data_folder, 'output/models/model_' +
                                                     helpers.get_model_timestamp()), verbose=True)

        return tb_viz_cb, tb_logs_cb, model_saver_cb

    def train(self, train_dataset, validate_dataset, epochs, batch_size = 1, threads = cpu_count(), use_cuda=False):

        train_loader = DataLoader(train_dataset, batch_size,
                              sampler=RandomSampler(train_dataset),
                              num_workers=threads,
                              pin_memory=use_cuda)
        
        valid_loader = DataLoader(validate_dataset, batch_size,
                              sampler=SequentialSampler(validate_dataset),
                              num_workers=threads,
                              pin_memory=use_cuda)
        
        print("Training on {} samples and validating on {} samples "
          .format(len(train_loader.dataset), len(valid_loader.dataset)))

        
        self.classifier.train(train_loader, valid_loader, epochs,
                     callbacks=self.get_callbacks())

    def restore_model(self,path):
        
        self.classifier.restore_model(path)

    def restore_last_model(self):
        model_folder = os.path.join(os.environ.get('KITCAR_REPO_PATH'), 'kitcar-simulation-data','output/models')
        model_file = sorted(os.listdir(model_folder))[-1]
        print('Restoring from ' + model_file)
        self.classifier.restore_model(os.path.join(model_folder,model_file))

    def predict(self, inp):
        return self.net(inp)[0]

class DatabaseTrainer(Trainer):
    def __init__(self, network, database_connector, classifier, img_size = [160,320]):
        self.image_size = img_size
        self.db_connector = database_connector
        
        super(DatabaseTrainer, self).__init__(network,classifier)


    def load_training_datasets(self, dataset_name, max_inputs, validation_factor):
        db_images = self.db_connector.load_all(dataset_name=dataset_name, random=True,max_count=max_inputs)
        file_count = len(db_images)
        train_ds = ds.SegmentationImageDataset(segmentation_images=db_images[int(file_count*validation_factor):])
        valid_ds = ds.SegmentationImageDataset(segmentation_images=db_images[:int(file_count*validation_factor)])

        return train_ds,valid_ds

    def train(self, dataset_name, epochs=1, max_inputs=20, validation_factor = 0.2, batch_size = 1, threads = cpu_count(), use_cuda=False):
        train_ds,valid_ds = self.load_training_datasets(dataset_name= dataset_name,max_inputs=max_inputs,validation_factor=validation_factor)
        super(DatabaseTrainer, self).train(train_ds,valid_ds,epochs=epochs,batch_size=batch_size,threads=threads,use_cuda=use_cuda)

    def load_test_dataset(self, test_images):
        return ds.SegmentationImageDataset(segmentation_images=test_images)
        
    def predict_tests(self,test_images, threads = cpu_count(), use_cuda=False):
        predictions = []
        
        test_ds = self.load_test_dataset(test_images)
        self.net.eval()
        
        for i in range(0,test_ds.__len__()-1):
            img = test_ds.__getitem__(i)[0].unsqueeze(0)
            predictions.append(self.net(img)[0])

        return predictions
        

                            