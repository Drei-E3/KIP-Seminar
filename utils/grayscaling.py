import os
import cv2
import numpy as np
from pathlib import Path
import argparse

workingDirectory = Path.cwd()
print(workingDirectory)

default_folderRead = Path('read\PixelLabelData')
default_folderWrite = Path('write')
default_numClasses =  2
default_writeDirectory = workingDirectory / default_folderWrite
default_readDirectory = workingDirectory / default_folderRead


class Grayscaler:
    
    def __init__(self,
                 readDirectory =default_readDirectory,
                 writeDirectory =default_writeDirectory,
                 numClasses = default_numClasses):
        """
        Initializes the Grayscaler with directories for reading and writing images and the number of grayscale classes.
        
        Args:
            folderRead (str): Path to the directory containing input images.
            folderWrite (str): Path to the directory where processed images will be saved.
            numClasses (int): Number of grayscale classes to categorize the pixels into.
        """
        self.readDirectory = Path(readDirectory)
        self.writeDirectory = Path(writeDirectory)
        if not self.writeDirectory.exists():
            os.makedirs(self.writeDirectory)
        self.img_name_list = sorted(os.listdir(self.readDirectory))
        self.numClasses = numClasses
        self.uniqueColors = set()
        self.thresholds = None
        self.colorClasses = None
    
    def call_size(self):
        """
        Prints the number of images in the dataset.
        """
        print('Dataset size: ', len(self.img_name_list))
        return len(self.img_name_list)
    
    def parse_color(self):
        """
        Parses unique grayscale values from the images in the read directory.
        """
        for img_name in self.img_name_list:
            img_path = os.path.join(self.readDirectory, img_name)
            img = cv2.imread(img_path, flags=0)  # Open image in greyscale mode
            self.uniqueColors |= set(np.unique(img))
        print("Found Colors (unique greyscale values [0..255]): ", len(self.uniqueColors))
        return len(self.uniqueColors)
    
    def scale_gray(self):
        """
        Categorizes the grayscale values using thresholds and generates new grayscale classes.
        """
        # Categorize using thresholds
        minimum = min(self.uniqueColors)
        maximum = max(self.uniqueColors)
        print('Minimum Threshold: ' + str(minimum))
        print('Maximum Threshold: ' + str(maximum))
        # Generate linearly distributed thresholds
        self.thresholds = np.linspace(start=minimum, stop=maximum + 1, num=self.numClasses + 1)
        # Generate linearly distributed classes [0..255]
        self.colorClasses = np.linspace(start=0, stop=255, num=self.numClasses)
        print("New greyscale values: ", self.colorClasses)
        return self.colorClasses

    def color2grey(self):
        """
        Applies the thresholds to the images and saves the processed images to the write directory.
        
        :param writeDirectory: Directory to write processed images to.
        """
        # Apply thresholds on masks and export to desktop
        grey_imgs = []
        for img_name in self.img_name_list:
            img_path = os.path.join(self.readDirectory, img_name)
            img = cv2.imread(img_path, flags=0)  # Open image in greyscale mode
            output_img = np.zeros_like(img)

            for i in range(self.numClasses):
                lower_bound = self.thresholds[i]
                upper_bound = self.thresholds[i + 1]
                mask = cv2.inRange(img, lower_bound, upper_bound)
                output_img[mask > 0] = self.colorClasses[i]

            output_path = os.path.join(self.writeDirectory, ''+img_name.rsplit('.', 1)[0]+'_masked.png')
            grey_imgs.append(output_img)
            cv2.imwrite(output_path, output_img)

        print("Images processed and saved to:", self.writeDirectory)
        return grey_imgs
        
    def run(self):
        """
        Initializes the processing by calling the size, color parsing, grayscale scaling, and image processing methods.
        """
        self.call_size()
        self.parse_color()
        self.scale_gray()
        return self.color2grey()






def parse_args():
    """Parses command line arguments for grayscale conversion."""
    parser = argparse.ArgumentParser(description="Convert images to scaled grayscale images based on specified classes.")
    parser.add_argument("--interactive_off", action="store_true", help="Run script in interactive mode.",default=False)
    parser.add_argument("--readDirectory", type=str, default="read/PixelLabelData", help="Directory path to read images from.")
    parser.add_argument("--writeDirectory", type=str, default="write", help="Directory path to write processed images to.")
    parser.add_argument("--numClasses", type=int, default=2, help="Number of grayscale classes for image categorization.")
    args = parser.parse_args()
    if not args.interactive_off:
        args = parse_interactive_args()
    return args

def parse_interactive_args():
    """Parses command line arguments interactively."""
    print("Augment images with specific transformations.")
    readDirectory = input("Directory path to the input (labeled) images: (default: read/PixelLabelData") or "read/PixelLabelData"
    writeDirectory = input("Directory path for saving scaled images (default: writeDirectory): ") or "writeDirectory"
    numClasses = input("Number of grayscale classes for image categorization (default 2): ") or "2"
    class Args:
        def __init__(self):
            self.readDirectory = readDirectory
            self.writeDirectory = writeDirectory
            self.numClasses = numClasses
    return Args()



if __name__ == "__main__":
    args = parse_args()
    grayscaler = Grayscaler(args.readDirectory, args.writeDirectory, args.numClasses)
    grayscaler.run()
    if args.interactive:
        to_exit = False
        while not to_exit:
            print('current status of Augumentator:')
            print(f"1. input folder of (labeled) images: {args.readDirectory}")
            print(f"2. output Directory path for saving scaled images: {args.writeDirectory}")
            print(f"3. Number of grayscale classes for image categorization : {args.numClasses}")
            print("4. excute")
            print("5. exit")
            
            change = int(input("Enter number to select setting you want to change: ") or 0)
            if change == 0:
                print("no input, enter 5 to exit if you want stop")
            elif change == 1:
                args.in_folder = input("Directory path to the input (labeled) images:") or args.readDirectory
            elif change == 2:
                args.out_folder = input("Directory path for saving scaled images") or args.writeDirectory
            elif change == 3:
                args.crop_width = input("Number of grayscale classes for image categorization") or args.numClasses
            elif change == 4:
                execute = bool(int(input("do you really want to run cropping with current setting? 1 for yes and 0 for no: ") or 0))
                if execute:
                    grayscaler = Grayscaler(args.readDirectory, args.writeDirectory, args.numClasses)
                    grayscaler.run()
            elif change == 5:
                to_exit = 1