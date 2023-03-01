
from os import listdir
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
from mrcnn.utils import Dataset
import matplotlib.pyplot as plt
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes

from mrcnn.config import Config
from mrcnn.model import MaskRCNN


class FruitsDataset(Dataset):
    # loading dataset 
    def load_dataset(self, dataset_dir, is_train=True):
        # define classes or labels
        self.add_class("dataset", 1, "apple")
        self.add_class("dataset", 2, "banana")
        self.add_class("dataset", 3, "orange")
        
        # define data locations
        images_dir = dataset_dir + '/images/'
        annotations_dir = dataset_dir + '/annots/'
       
             
	# Find Images
        for filename in listdir(images_dir):
            print(filename)
	    # Extract image id
            image_id = filename[:-4]
	    #print('IMAGE ID: ',image_id)
			
	    # Creating train dataset
            if is_train and int(image_id) >= 250:
                continue
	    # Creating test dataset
            if not is_train and int(image_id) < 250:
                continue
            img_path = images_dir + filename
            ann_path = annotations_dir + image_id + '.xml'
	    # Add to dataset
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path, class_ids = [0,1,2,3])


    # Extract bounding boxes from .xml file
    def extract_boxes(self, filename):
	# Load and parse the file
        tree = ElementTree.parse(filename)
	# Getting the root of the document
        root = tree.getroot()
	# Extract each bounding box
        boxes = list()
        for box in root.findall('.//object'):
            name = box.find('name').text  
            xmin = int(box.find('./bndbox/xmin').text)
            ymin = int(box.find('./bndbox/ymin').text)
            xmax = int(box.find('./bndbox/xmax').text)
            ymax = int(box.find('./bndbox/ymax').text)
            coors = [xmin, ymin, xmax, ymax, name]
            boxes.append(coors)

	# Extract image dimensions
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height

    # Load the masks for an image
    def load_mask(self, image_id):
	# Get details of image
        info = self.image_info[image_id]
	# Define box file location
        path = info['annotation']
        #return info, path
        
        
	# load XML
        boxes, w, h = self.extract_boxes(path)
	# create one array for all masks, each on a different channel
        masks = zeros([h, w, len(boxes)], dtype='uint8')
	# create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            
            
            # box[4] will have the name of the class 
            if (box[4] == 'apple'):
                masks[row_s:row_e, col_s:col_e, i] = 1
                class_ids.append(self.class_names.index('apple'))
            elif(box[4] == 'banana'):
                masks[row_s:row_e, col_s:col_e, i] = 2
                class_ids.append(self.class_names.index('banana')) 
            elif(box[4] == 'orange'):
                masks[row_s:row_e, col_s:col_e, i] = 3
                class_ids.append(self.class_names.index('orange'))
          
        return masks, asarray(class_ids, dtype='int32')
        

    # Load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

# Train set
dataset_dir='Dataset'

train_set = FruitsDataset()
train_set.load_dataset(dataset_dir, is_train=True)
train_set.prepare()
print('Amount of training data: %d' % len(train_set.image_ids))

# Test set
test_set = FruitsDataset()
test_set.load_dataset(dataset_dir, is_train=False)
test_set.prepare()
print('Amount of testing data: %d' % len(test_set.image_ids))




# Define a configuration for the model
class FruitsConfig(Config):
	# define the name of the configuration
	NAME = "fruits_cfg"
	# number of classes (background + 3 fruits)
	NUM_CLASSES = 1 + 3
	# number of training steps per epoch
	STEPS_PER_EPOCH = 100
    
    
# prepare config
config = FruitsConfig()
config.display()

# define the model
model = MaskRCNN(mode='training', model_dir="./", config=config)
# load weights (mscoco) and exclude the output layers
model.load_weights("mask_rcnn_coco.h5", by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

# train weights (output layers or 'heads')
model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=20, layers='heads')

model_path = 'Fruits_MaskRCNN_trained.h5'
model.keras_model.save_weights(model_path)

