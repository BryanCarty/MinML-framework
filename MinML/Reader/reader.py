import pandas as p
import random
import numpy as np
from PIL import Image
import threading
import time
import queue

class Reader():
    def __init__(self, csv_file, batch_size, num_epochs, normalize_pixels=None, one_hot_encoding=None):
        self.normalize_pixels = normalize_pixels
        self.one_hot_encoding = one_hot_encoding
        self.csv_file = csv_file
        self.num_samples = len(csv_file)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.batch_sequences = [random.sample(range(self.num_samples), self.num_samples) for _ in range(num_epochs)]
        self.batch_sequences = [item for sublist in self.batch_sequences for item in sublist]
        self.batch_sequences = [self.batch_sequences[i:i+self.batch_size] for i in range(0, len(self.batch_sequences), self.batch_size)]
        if (self.num_samples * 5)%self.batch_size != 0:
            self.batch_sequences = self.batch_sequences[:-1]
        self.next_batches = queue.Queue(maxsize=5)
        self.batches_left = len(self.batch_sequences)

        
        # Store first five batches in memory
        for i in range(5):
            batch_items = self.batch_sequences.pop() # batch indexes
            acquired_batch_items = [] 
            for val in batch_items: # going through indexes
                batch_item = [] # singular acquired batch item
                img_path = self.csv_file.loc[val, 'pth']
                img_label = self.csv_file.loc[val, 'label']
                if self.one_hot_encoding != None:
                    img_label = [1 if cls == img_label else 0 for cls in self.one_hot_encoding]
                img = np.array(Image.open("archive/"+img_path))
                if self.normalize_pixels != None:
                    img = img/255
                batch_item.append(img)
                batch_item.append(img_label)
                acquired_batch_items.append(batch_item)
            self.next_batches.put(acquired_batch_items)
        self.process_thread = threading.Thread(target=self.process_batch).start()
        
    def process_batch(self): # if (there's ones to move over and they're currently not in process)
        while True:
            if len(self.batch_sequences) > 0:
                batch_item = self.batch_sequences.pop() # batch index
                acquired_batch_item = [] 
                for val in batch_item: # going through indexes
                    batch_item = [] # singular acquired batch item
                    img_path = self.csv_file.loc[val, 'pth']
                    img_label = self.csv_file.loc[val, 'label']
                    if self.one_hot_encoding != None:
                        img_label = [1 if cls == img_label else 0 for cls in self.one_hot_encoding]
                    img = np.array(Image.open("archive/"+img_path))
                    if self.normalize_pixels != None:
                        img = img/255                    
                    batch_item.append(img)
                    batch_item.append(img_label)
                    acquired_batch_item.append(batch_item)
                self.next_batches.put(acquired_batch_item)
            else:
                return

    def next(self):
        if self.batches_left > 0:
            next_batch = self.next_batches.get()
            x = np.array([item[0] for item in next_batch])  # Shape: 128 x 96 x 96 x 3
            y = np.array([item[1:2] for item in next_batch])
            self.batches_left -= 1
            return x, y
        return False






'''
d = p.read_csv('archive/labels.csv')
reader = Reader(d, 128, 5)
while True:
    next_batch = reader.next()
    if not next_batch:
        break
    time.sleep(0.01)
'''