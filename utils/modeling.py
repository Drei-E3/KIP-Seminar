import argparse
import numpy as np
import os
from pathlib import Path
import cv2
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, concatenate
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split, ParameterGrid
from skimage.transform import resize


from keras.metrics import Accuracy  
from keras.callbacks import ModelCheckpoint 



def parse_list(value_str):
    """Parses a string of numbers separated by commas into a list of floats or integers."""
    values = value_str.split(',')
    return [float(v) if '.' in v else int(v) for v in values]

class WearDetector(Model):
    """A convolutional neural network model for detecting wear in images using TensorFlow.

    This class inherits from `tf.keras.Model` and setups a simple CNN model architecture
    for binary classification tasks.

    Attributes:
        filters (int): Number of filters in the first convolutional layer.
        dropout_rate (float): Dropout rate for the dropout layers.
        IMG_HEIGHT (int): Height of the input images.
        IMG_WIDTH (int): Width of the input images.
        IMG_CHANNELS (int): Number of channels in the input images.
        model (tf.keras.Model): The underlying Keras model initialized during the build process.
    """

    def __init__(self, filters=16, dropout_rate=0.1, target_size=(512, 512) , IMG_CHANNELS=1):
        super(WearDetector, self).__init__()
        self.filters = filters
        self.dropout_rate = dropout_rate
        self.IMG_HEIGHT = target_size[0]
        self.IMG_WIDTH = target_size[1]
        self.IMG_CHANNELS = IMG_CHANNELS
        self.build_model()


    def build_model(self):
        
        inputs = Input((self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS), name='input_layer')
        
        c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
        c1 = Dropout(0.1)(c1)
        c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
        p1 = MaxPooling2D((2, 2))(c1)

        c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
        c2 = Dropout(0.1)(c2)
        c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
        p2 = MaxPooling2D((2, 2))(c2)
            
        c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
        c3 = Dropout(0.2)(c3)
        c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
        p3 = MaxPooling2D((2, 2))(c3)
            
        c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
        c4 = Dropout(0.2)(c4)
        c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
        p4 = MaxPooling2D(pool_size=(2, 2))(c4)
            
        c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
        c5 = Dropout(0.3)(c5)
        c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

        #Expansive path 
        #DAS IS DECODER STRUKTUR
        u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
        c6 = Dropout(0.2)(c6)
        c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
            
        u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
        c7 = Dropout(0.2)(c7)
        c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
            
        u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
        c8 = Dropout(0.1)(c8)
        c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
            
        u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1], axis=3)
        c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
        c9 = Dropout(0.1)(c9)
        c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
            
        outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
        self.model = Model(inputs=[inputs], outputs=[outputs])

    def call(self, inputs):
        return self.model(inputs)

    def compile(self,optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']):
        super().compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
    def load_data(self,data_path,target_size=(512, 512)):
        
        images_dir = data_path['images']
        masks_dir = data_path['masks']
        images = []
        masks = []

        # Load images and corresponding masks
        for filename in os.listdir(images_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png','tiff')):
                img_path = os.path.join(images_dir, filename)
                split_filename = filename.split('.')
                mask_path = os.path.join(masks_dir, split_filename[0] + '_masked.' + split_filename[1]) 
                # Read the image and mask
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  
                # resize, and optionally convert to TIFF in memory
                if img is not None and mask is not None:
                    img = resize(img, target_size)
                    mask = resize(mask, target_size)
                images.append(np.array(img))
                masks.append(np.array(mask))
                

        # Normalize and prepare dataset
        self.X = np.array(images, dtype='float32') / 255.0  # normalize images
        self.y = np.array(masks, dtype='float32') / 255.0  # normalize masks if needed
        
        self.IMG_HEIGHT = self.X[0].shape[1]
        self.IMG_WIDTH = self.X[0].shape[0]

    def train(self, loss='binary_crossentropy', lr= 0.01,epochs=10, batch_size=10,test_size=0.2, verbose=1, eval=True):
        """Trains the model on provided data, splits it into training and testing datasets.
        
        Args:
            data_path (dict): Dictionary containing paths to directories of images and masks.
            train_test_split_rate (float): Fraction of the dataset to be used as test data.
            epochs (int): Number of epochs to train the model.
            batch_size (int): Number of samples per batch of computation.
            verbose (int): Verbosity mode.
        """
        # Split dataset
        X_train, X_test,y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)      
        
        # Train the model
        optimizer=Adam(learning_rate=lr)
        self.compile(optimizer=optimizer, loss=loss, metrics=self.model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy']))
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_split=0.2)
        print("Training complete with final accuracy: {:.2f}%".format(history.history['accuracy'][-1] * 100))

        # Optionally, evaluate on test set
        if eval:
            self.evaluate(X_test, y_test)

    def evaluate(self, X,y, verbose=1):
        """Evaluates the model on provided testing data and prints the accuracy."""
        results = self.model.evaluate(X, y, verbose=verbose)
        print(f"Evaluation results - Loss: {results[0]}, Accuracy: {results[1]*100:.2f}%")
    '''
    def predict(self, images_dir, verbose=0):
        images = []
        # Load images and corresponding masks
        for filename in os.listdir(images_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png','tiff')):
                img_path = os.path.join(images_dir, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                # resize
                img = cv2.resize(img, (self.IMG_WIDTH, self.IMG_HEIGHT))

                images.append(np.array(img, dtype=np.uint8))
        return self.model.predict(images, verbose=verbose)
    ''' 
        

    def predict(self, images_dir, verbose=0,target_size=(512, 512),save_path=False,save_in_tiff=False):
        images = []
        names = []
        # Load images and corresponding masks
        for filename in os.listdir(images_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png','tiff')):
                img_path = os.path.join(images_dir, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                # resize
                img = resize(img, target_size)

                images.append(np.array(img, dtype=np.uint8))
                names.append(filename)
        images= np.array(images, dtype='float32') / 255.0 
        # Convert list of images to 4D numpy array
        images = np.expand_dims(np.array(images), axis=-1)
        result = self.model.predict(images, verbose=verbose)
        if save_path:
            img_extension = ".tiff" if save_in_tiff else ""
            if not os.path.exists(Path(save_path)):
                os.makedirs(save_path)
            for id, img in enumerate(result):
                img_filename = "prediction"  +"_"+ names[id] + img_extension
                cv2.imwrite(str(Path(save_path) / img_filename), img)
        return result
    
    def predict_for_pics(self, pics, verbose=0):
        images = []
        # Load images and corresponding masks
        for pic in pics:
                images.append(np.array(pic, dtype=np.uint8)* 255.0 )
        
        # Convert list of images to 4D numpy array
        images = np.expand_dims(np.array(images), axis=-1)
        
        return self.model.predict(images, verbose=verbose)

    def save_model(self, file_path='wear_detector_model.h5'):
        self.model.save(file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, file_path):
        self.model = tf.keras.models.load_model(file_path)
        self.model.compile()
        print(f"Model loaded from {file_path}")

    def grid_search_train(self,param_grid, epochs=10, batch_size=10,test_size=0.2):
        """Trains the model using GridSearchCV to find the best hyperparameters."""
        X = self.X
        y = self.y
        param_grid = param_grid
        grid = ParameterGrid(param_grid)
        best_score = float('inf')
        best_params = 0
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size)
        for params in grid:
            print("epochs|batch size|learning rate| test split rate")
            print(params['epochs'], params['batch_size'],params['lr'],test_size)
            score = self.train( epochs=params['epochs'], batch_size=params['batch_size'],lr=params['lr'],test_size=0.2)
            if score < best_score:
                best_score = score
                best_params = params
                self.save_model(f"best_model_lr_{params['lr']}_batch_{params['batch_size']}_epoch_{params['epochs']}.h5")
        print("Best Loss:", best_score)
        print("Best Hyperparameters:", best_params)
        
       
       
       
       
       
       
       
       
        
def parse_args():
    parser = argparse.ArgumentParser(description='Manage the WearDetector model.')
    parser.add_argument("--interactive_off", action="store_true", help="Run script in interactive mode (default interactive is on).",default=False)
    parser.add_argument('--action', type=str, default='train', choices=['train', 'load', 'predict', 'evaluate', 'grid_search'],
                        help='Action to perform: train, load, predict, evaluate, or grid_search.')
    parser.add_argument('--model_path', type=str, default='wear_detector_model.h5', help='Path to model file for loading or saving.')
    parser.add_argument('--image_path', type=str, help='Path to the dataset file.')
    parser.add_argument('--mask_path', type=str, help='Path to the dataset file.')
    parser.add_argument('--lr', type=parse_list, help='Comma-separated list of filter counts for grid search.')
    parser.add_argument('--batch_size', type=parse_list, help='Comma-separated list of dropout rates for grid search.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training or grid search.')
    parser.add_argument('--train_test_split_rate', type=float, default=0.2, help='Ratio of data split for testing.')
    
    args = parser.parse_args()
    if not args.interactive_off:
        args = parse_interactive_args()
    return args



def main_menu():
    args = parse_args()
    model = WearDetector()
    
    data_path = {"images": args.image_path,
                 "masks": args.mask_path}
    model.load_data(data_path)
    if args.action == 'load':
        model.load_model(args.model_path)
        choice = input("Do you want to (r)etrain, (p)redict, or (e)valuate the model? [r/p/e]: ")
        if choice.lower() == 'r':
            model.train(args.data_path, args.train_test_split_rate,args.epochs, args.batch_size)
            model.save_model(args.model_path)
        elif choice.lower() == 'p':
            input_img = input("enter the image folder you want to predict: ")
            save_path = input("enter the folder path you want to save the prediction: ")
            predictions = model.predict(input_img,save_path=save_path)
            print(predictions)
        elif choice.lower() == 'e':
            model.evaluate(args.image_path, args.mask_path)
    elif args.action == 'train':
        model.train(args.data_path, args.train_test_split_rate, args.epochs, args.batch_size)
        model.save_model(args.model_path)
    elif args.action == 'grid_search':
        param_grid = {
            'lr': args.lr if args.lr else [0.01, 0.001],
            'batch_size': args.batch_size if args.batch_size else [16, 32],
            'epochs':args.epochs if args.epochs else [10 , 20]
        }
        model.grid_search_train( param_grid, test_size=args.test_size)
        

def parse_interactive_args():
    image_path = None
    mask_path = None
    lr = None
    epochs = None
    batch_size = None
    class Args:
        def __init__(self):
            self.image_path = image_path
            self.mask_path = mask_path
            self.lr = lr
            self.epochs = epochs
            self.batch_size = batch_size
            self.interactive = True
    return Args()

def main():
    args = parse_args()
    model = None
    if args.interactive:
        to_exit = False
        while not to_exit:
            print("select operation you want to apply:")
            choice = input("Do you want to (r)etrain, (p)redict, or (e)valuate the model or exi(t)? [r/p/e/t]: ")
            if choice.lower() == 'r':
                print("please enter the dataset to train")
                args.image_path = input("entery the path to the images: ")
                args.mask_path = input("entry the path to the target data (masks): ")
                args.test_size = input("entry test splite rate (default 0.2)") or "0.2"
                print("please set hyperparameters")
                args.lr = input("enter comma-separated list of filter counts for grid search (default 0.001): ") or "0.001"
                args.epochs = input("enter number of epochs for training or grid search (default 50)") or "50"
                args.batch_size = input("enter number of epochs for training or grid search (default 50)") or "16"

                
                param_grid = {
                'lr': args.lr if args.lr else [0.01, 0.001],
                'batch_size': args.batch_size if args.batch_size else [16, 32],
                'epochs':args.epochs if args.epochs else [10 , 20]
                }
                model = WearDetector()
                data_path = {"images": args.image_path,
                    "masks": args.mask_path}
                model.load_data(data_path)
                model.grid_search_train( param_grid, test_size=args.test_size)
                
            elif choice.lower() == 'p':
                if model is None : 
                    model_checkpoint = input("enter the dictionary of model you want to load: ")
                    model = WearDetector()
                    model.load_model(model_checkpoint)
                input_img = input("enter the image folder you want to predict: ")
                save_path = input("enter the folder path you want to save the prediction: ")
                predictions = model.predict(input_img,save_path=save_path)
                print("prediction is finished")
                
            elif choice.lower() == 'e':
                eva_imgs = input("enter the evalation images folder: ")
                eva_msks = input("enter the evalation masks folder: ")
                model.evaluate(eva_imgs, eva_msks)
            
            elif choice.lower() == 't':
                to_exit = 1
    else:
        model = WearDetector()
        
        data_path = {"images": args.image_path,
                    "masks": args.mask_path}
        model.load_data(data_path)
        if args.action == 'load':
            model.load_model(args.model_path)
            choice = input("Do you want to (r)etrain, (p)redict, or (e)valuate the model? [r/p/e]: ")
            if choice.lower() == 'r':
                model.train(args.data_path, args.train_test_split_rate,args.epochs, args.batch_size)
                model.save_model(args.model_path)
            elif choice.lower() == 'p':
                input_img = input("enter the image folder you want to predict: ")
                save_path = input("enter the folder path you want to save the prediction: ")
                predictions = model.predict(input_img,save_path=save_path)
                print(predictions)
            elif choice.lower() == 'e':
                model.evaluate(args.image_path, args.mask_path)
        elif args.action == 'train':
            model.train(args.data_path, args.train_test_split_rate, args.epochs, args.batch_size)
            model.save_model(args.model_path)
        elif args.action == 'grid_search':
            param_grid = {
                'lr': args.lr if args.lr else [0.01, 0.001],
                'batch_size': args.batch_size if args.batch_size else [16, 32],
                'epochs':args.epochs if args.epochs else [10 , 20]
            }
            model.grid_search_train( param_grid, test_size=args.test_size)
            


if __name__ == '__main__':
    main()