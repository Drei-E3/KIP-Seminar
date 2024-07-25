import cv2
import os
from pathlib import Path
import argparse


###working Directory
workingDirectory = Path.cwd()
print(workingDirectory)

# Input and output directories
folderImg = Path('int_img')
folderCroppedImg = ('out_cropped_img')

# default setting
default_in_folder = workingDirectory / folderImg
default_out_folder = workingDirectory / folderCroppedImg
default_crop_width = 1400
default_crop_height = 1840

class Cropper:
    """A class for cropping images to a specified size."""
    
    def __init__(self,
                 in_folder=default_in_folder,
                 out_folder= default_out_folder):
        """
        Initializes the Cropper with input and output directories.

        Args:
            in_folder (str): Path to the directory containing input images.
            out_folder (str): Path to the directory where cropped images will be saved.
        """
        self.in_folder = in_folder
        self.out_folder = out_folder
        if not os.path.exists(self.out_folder):
            os.makedirs(self.out_folder)
        self.crop_width = None
        self.crop_height = None
                    
    def reset_path(self,
                 in_folder,
                 out_folder):
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        self.in_folder = in_folder
        self.out_folder = out_folder
    
    
    def set_cropping_size(self,
                          crop_width=default_crop_width,
                          crop_height = default_crop_height):
        """
        Set the dimensions for cropping the images.

        Args:
            crop_width (int): The width of the crop area.
            crop_height (int): The height of the crop area.
        """
        self.crop_height= crop_height
        self.crop_width = crop_width
    
    def crop_save(self,save_in_tiff=False):
        """Crops each image in the input directory and saves them to the output directory."""
        for filename in os.listdir(self.in_folder): 
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                # Read the image
                img_path = os.path.join(self.in_folder, filename)
                img = cv2.imread(img_path)
                
                # Get dimensions of the image
                height, width = img.shape[:2]
                
                # Calculate the coordinates for cropping
                start_x = max(width // 2 - self.crop_width // 2, 0)
                end_x = min(start_x + self.crop_width, width)
                start_y = max(height // 2 - self.crop_height // 2, 0)
                end_y = min(start_y + self.crop_height, height)
                
                # Crop the image
                cropped_img = img[start_y:end_y, start_x:end_x]
                
                # Save the cropped image
                if not save_in_tiff:
                    out_path = os.path.join(self.out_folder, filename)
                    cv2.imwrite(out_path, cropped_img)
                else:
                    last_dot_index = filename.rfind(".")
                    out_path = os.path.join(self.out_folder, filename[:last_dot_index]+".tiff")
                    cv2.imwrite(out_path, cropped_img)
    
    def run(self,crop_width=default_crop_width,crop_height = default_crop_height,save_in_tiff=False):
        """Configures the cropper and initiates the cropping process."""
        self.set_cropping_size(crop_width, crop_height)
        self.crop_save(save_in_tiff)

def parse_args():
    """Parses command line arguments for cropping images."""
    parser = argparse.ArgumentParser(description="Crop images to a specified size and save them to a specified directory.")
    parser.add_argument("--interactive_off", action="store_true", help="Run script in interactive mode.",default=False)
    parser.add_argument("--in_folder", type=str, default="int_img", help="Directory path to the input images.")
    parser.add_argument("--out_folder", type=str, default="out_cropped_img", help="Directory path for saving cropped images.")
    parser.add_argument("--crop_width", type=int, default=1400, help="Width of the cropped image.")
    parser.add_argument("--crop_height", type=int, default=1840, help="Height of the cropped image.")
    parser.add_argument("--save_in_tiff", action="store_true", help="Flag to save the output in TIFF format instead of default PNG.")
        
    args = parser.parse_args()
    if not args.interactive_off:
        args = parse_interactive_args()
    return args


def parse_interactive_args():
    """Parses command line arguments interactively."""
    print("Augment images with specific transformations.")
    in_folder = input("Directory path to the input images: (default: int_img)") or "int_img"
    out_folder = input("Directory path for saving cropped images (default: out_cropped_img): ") or "out_cropped_img"
    crop_width = input("Enter Width of the cropped image (default 1400): ") or "1400"
    crop_height = input("Height of the cropped image (default: 1840): ") or "1840"
    save_in_tiff = input("save in which format,default in the same format as orignal (default: False): ") or False
    class Args:
        def __init__(self):
            self.in_folder = in_folder
            self.out_folder = out_folder
            self.crop_width = crop_width
            self.crop_height = crop_height
            self.save_in_tiff = save_in_tiff
            self.interactive = True
    return Args()

if __name__ == "__main__":
    args = parse_args()
    cropper = Cropper(args.in_folder, args.out_folder)
    cropper.run(args.crop_width, args.crop_height, args.save_in_tiff)
    if args.interactive:
        to_exit = False
        while not to_exit:
            print('current status of Augumentator:')
            print(f"1. input folder of images: {args.in_folder}")
            print(f"2. output Directory path for saving cropped images: {args.out_folder}")
            print(f"3. Width of the cropped image: {args.crop_width}")
            print(f"4. Height of the cropped image: {args.crop_height}")
            print(f"5. is output format in tiff?: {args.save_in_tiff}")
            print("6. excute")
            print("7. exit")
            
            change = int(input("Enter number to select setting you want to change: ") or 0)
            if change == 0:
                print("no input, enter 7 to exit if you want stop")
            elif change == 1:
                args.in_folder = input("Directory path to the input images: (default: int_img)") or args.in_folder
            elif change == 2:
                args.out_folder = input("Directory path for saving cropped images: (default: out_img)") or args.out_folder
            elif change == 3:
                args.crop_width = int(input("Width of the cropped image: (default: 100)") or args.crop_width)
            elif change == 4:
                args.crop_height = int(input("Height of the cropped image: (default: 100)") or args.crop_height)
            elif change == 5:
                args.save_in_tiff = bool(input("Is output format in tiff? (default: False)") or args.save_in_tiff)
            elif change == 6:
                execute = bool(int(input("do you really want to run cropping with current setting? 1 for yes and 0 for no: ") or 0))
                if execute:
                    cropper = Cropper(args.in_folder, args.out_folder)
                    cropper.run(args.crop_width, args.crop_height, args.save_in_tiff)
            elif change == 7:
                to_exit = 1