import cv2
import os
from pathlib import Path
from scipy.ndimage import rotate,shift
import random
import argparse
import os

# working Directory
workingDirectory = Path.cwd()
print(workingDirectory)

# default path 
## original data paths
folderImg = Path('Ressources/images')
folderMsk = Path('Ressources/masks')
default_images_path = workingDirectory / folderImg
default_masks_path = workingDirectory / folderMsk
## paths for saving folders for augmented data
folderAugImg = Path('Ressources/img_augmented_path')
folderAugMsk = Path('Ressources/msk_augmented_path')
default_images_aug_path = workingDirectory / folderAugImg
default_masks_aug_path = workingDirectory / folderAugMsk
default_seed = 42

class Augumentator:
    """A class for augmenting images with various transformations such as rotations, flips, and translations."""
    
    def __init__(self,
                 img_path=default_images_path,
                 msk_path=default_masks_path,
                 seed = default_seed):
        """
        Initializes the Augmentator with paths for images and masks, and a seed for randomness.

        Args:
            img_path (Path): The file path to the directory containing images.
            msk_path (Path): The file path to the directory containing masks.
            seed (int): Seed value for random number generator to ensure reproducibility.
        """
        self.img_path = Path(img_path)
        self.msk_path = Path(msk_path)
        self.images=[] 
        self.images_name=[]
        self.masks=[]
        self.masks_name=[]
        self.counter= 0
        
        self.seed = seed
        random.seed(self.seed)
        self.inplace=None
        
        # load pictures and its masks
        if self.img_path.exists():
            for im in os.listdir(self.img_path):
                if im.endswith(('.jpg', '.jpeg', '.png')):
                    image = cv2.imread(str(self.img_path / im))
                    if image is not None:
                        self.images.append(image)
                        self.images_name.append(im)
                
        if self.msk_path.exists():
            for msk in os.listdir(self.msk_path):
                if msk.endswith(('.jpg', '.jpeg', '.png')):
                    mask = cv2.imread(str(self.msk_path / msk))
                    if mask is not None:
                        self.masks.append(mask)
                        self.masks_name.append(msk)

    def global_inplace(self,on_off):
        self.inplace = bool(on_off)
        
    def set_seed(self,seed):
        """Set a new seed for random number generation."""
        self.seed = seed
        random.seed(self.seed)

    def rotation(self,angle_range=(-30, 30),inplace=False):
        """Rotate images randomly within a specified angle range."""
        if self.inplace: inplace =self.inplace
        
        r_img = []
        r_msk = []
        angle = random.randint(*angle_range)
        for img in self.images:
            rotated_img = rotate(img, angle=angle, reshape=False, mode='nearest')
            r_img.append(rotated_img)
        for msk in self.masks:
            rotated_msk = rotate(msk, angle=angle, reshape=False, mode='nearest')
            r_msk.append(rotated_msk)  
        if inplace:
            self.images,self.masks = r_img,r_msk          
        return r_img,r_msk
    

    def h_flip(self,inplace=False):
        """Flip images horizontally."""
        if self.inplace: inplace =self.inplace
        
        hflipped_img = [cv2.flip(img, 1) for img in self.images]
        hflipped_msk = [cv2.flip(msk, 1) for msk in self.masks]
        if inplace:
            self.images,self.masks = hflipped_img,hflipped_msk
        return hflipped_img,hflipped_msk

    def v_flip(self,inplace=False):
        """Flip images vertically."""
        if self.inplace: inplace =self.inplace
        
        vflipped_img = [cv2.flip(img, 0) for img in self.images]
        vflipped_msk = [cv2.flip(msk, 0) for msk in self.masks]
        if inplace:
            self.images,self.masks = vflipped_img,vflipped_msk
        return vflipped_img,vflipped_msk

    def h_transl(self, pixel_range=(-20, 20),inplace=False):
        """Translate images horizontally within a specified pixel range. default values are -20 to 20"""
        if self.inplace: inplace =self.inplace
        
        htranslated_img = []
        htranslated_msk = []
        pixels = random.randint(*pixel_range)
        for img in self.images:
            translated_img = shift(img, [0, pixels, 0])
            htranslated_img.append(translated_img)
        for msk in self.masks:
            translated_msk = shift(msk, [0, pixels, 0])
            htranslated_msk.append(translated_msk)
        if inplace:
            self.images,self.masks = htranslated_img,htranslated_msk
        return htranslated_img,htranslated_msk

    def v_transl(self, pixel_range=(-20, 20),inplace=False):
        """Translate images vertically within a specified pixel range. default values are -20 to 20"""
        if self.inplace: inplace =self.inplace
        
        vtranslated_img = []
        vtranslated_msk = []
        pixels = random.randint(*pixel_range)
        for img in self.images:
            translated_img = shift(img, [pixels, 0, 0])
            vtranslated_img.append(translated_img)
        for msk in self.masks:
            translated_msk = shift(msk, [0, pixels, 0])
            vtranslated_msk.append(translated_msk)
        if inplace:
            self.images,self.masks = vtranslated_img,vtranslated_msk
        return vtranslated_img,vtranslated_msk
      
    def save(self,
              img_path=default_images_aug_path,
              mask_path=default_masks_aug_path,
              save_in_tiff=False):
        """Save the augmented images and masks in the specified directory."""
        
        
        """
        # old codes
        if not save_in_tiff:
            for id in range(len(self.masks_name)):
                out_path = os.path.join(mask_path, self.msk_name[id])
                cv2.imwrite(out_path, self.masks[id])
            for id in range(len(self.images_name_name)):
                out_path = os.path.join(img_path, self.images_name_name[id])
                cv2.imwrite(out_path, self.images[id])
        else:
            for id in range(len(self.masks_name)):
                last_dot_index = self.msk_name[id].rfind(".")
                out_path = os.path.join(mask_path, self.msk_name[id][:last_dot_index]+".tiff")
                cv2.imwrite(out_path, self.masks[id])
            for id in range(len(self.images_name_name)):
                last_dot_index = self.images_name_name[id].rfind(".")
                out_path = os.path.join(mask_path, self.images_name_name[id][:last_dot_index]+".tiff")
                cv2.imwrite(out_path, self.images[id])
        """
        if self.inplace: inplace =self.inplace
        
        self.counter += 1
        img_extension = ".tiff" if save_in_tiff else ".png"
        if not os.path.exists(Path(img_path)):
            os.makedirs(img_path)
        if not os.path.exists(Path(mask_path)):  
            os.makedirs(mask_path)
        
        for id, img in enumerate(self.images):
            img_filename = "aug_" +str(self.counter) +"_"+ self.images_name[id].rsplit('.', 1)[0] + img_extension
            cv2.imwrite(str(Path(img_path) / img_filename), img)
        for id, msk in enumerate(self.masks):
            msk_filename = "aug_" +str(self.counter) +"_"+self.masks_name[id].rsplit('.', 1)[0] + img_extension
            cv2.imwrite(str(Path(mask_path) / msk_filename), msk)


def parse_args():
    """Parses command line arguments using argparse or interactively based on user choice."""
    parser = argparse.ArgumentParser(description="Augment images with specific transformations.")
    parser.add_argument("--interactive", action="store_true", help="Run script in interactive mode.",default=True)
    parser.add_argument("--img_path", type=str, default="Ressources/images", help="Directory path to the image files.")
    parser.add_argument("--msk_path", type=str, default="Ressources/masks", help="Directory path to the mask files.")
    parser.add_argument("--img_aug_path", type=str, default="Ressources/img_augmented_path", help="Directory path for saving augmented images.")
    parser.add_argument("--msk_aug_path", type=str, default="Ressources/msk_augmented_path", help="Directory path for saving augmented masks.")
    parser.add_argument("--save_in_tiff", type=bool, default=False, help="save in which format,default in the same format as orignal")
    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generator.")
    parser.add_argument("--processes", type=str, choices=['rotation', 'h_flip', 'v_flip', 'h_transl', 'v_transl'],default=None,
                        help="list of numbers to apply augmentation which separated by + (choices: 1:rotation, 2:h_flip, 3:v_flip, 4:h_transl, 5:v_transl.")   
    args = parser.parse_args()
    if args.interactive:
        args = parse_interactive_args()
    return args

def parse_interactive_args():
    """Parses command line arguments interactively."""
    print("Augment images with specific transformations.")
    img_path = input("Enter directory path to the image files (default: Ressources/images): ") or "Ressources/images"
    msk_path = input("Enter directory path to the mask files (default: Ressources/masks): ") or "Ressources/masks"
    img_aug_path = input("Enter directory path for saving augmented images (default: Ressources/img_augmented_path): ") or "Ressources/img_augmented_path"
    msk_aug_path = input("Enter directory path for saving augmented masks (default: Ressources/msk_augmented_path): ") or "Ressources/msk_augmented_path"
    seed = int(input("Enter seed for random number generator (default: 42): ") or 42)
    processes = input("Enter list of numbers to apply augmentation and separated numbers with + (choices: 1:rotation, 2:h_flip, 3:v_flip, 4:h_transl, 5:v_transl): ") or None
    save_in_tiff = input("save in which format,default in the same format as orignal (default: False): ") or False
    class Args:
        def __init__(self):
            self.img_path = img_path
            self.msk_path = msk_path
            self.img_aug_path = img_aug_path
            self.msk_aug_path = msk_aug_path
            self.seed = seed
            self.processes = processes
            self.save_in_tiff = save_in_tiff
            self.interactive = True
    
    return Args()


def main():
    def process(args,inplace=False):
        for process in args.processes.split('+'):
            if process is not None and process != "":
                # Process selected augmentations
                if process == '1':
                    rotated_images = augmenter.rotation(inplace=inplace)
                elif process == '2':
                    h_flipped_images = augmenter.h_flip(inplace=inplace)
                elif process == '3':
                    v_flipped_images = augmenter.v_flip(inplace=inplace)
                elif process == '4':
                    h_translated_images = augmenter.h_transl(inplace=inplace)
                elif process == '5':
                    v_translated_images = augmenter.v_transl(inplace=inplace)
                    
    args = parse_args()
    num2op = {"1":"rotation", "2":"h_flip", "3":"v_flip", "4":"h_transl", "5":"v_transl"}
    augmenter = Augumentator(args.img_path, args.msk_path, args.seed)
    augmenter.global_inplace(True)
    process(args)
    augmenter.save(img_path=args.img_aug_path,mask_path=args.msk_aug_path,save_in_tiff=args.save_in_tiff)
    
    if args.interactive:
        to_exit = False
        processes = []
        if args.processes != "" and args.processes is not None: 
            processes = [num2op[num] for num in args.processes.split('+')] 
        
        while(not to_exit):
            print('current status of Augumentator:')
            print(f"1. input folder of images: {args.img_path}")
            print(f"2. input folder of masks: {args.msk_path}")
            print(f"3. augumentation value (random seed): {args.seed}")
            print(f"4. output folder of images with augumentation: {args.img_aug_path}")
            print(f"5. output folder of masks with augumentation: {args.msk_aug_path}")
            print(f"6. is output format in tiff?: {args.save_in_tiff}")
            print(f"operations already processed (Order and times make difference):")
            print(processes)
            print("7. new augumentation operation")
            print("8. run")
            print("9. exit")
            
            
            change = int(input("Enter number to select setting you want to change: ") or 0)
            if change == 1:
                args.img_path = input("Enter directory path to the image files (default: Ressources/images): ") or args.img_path
            elif change == 2:
                args.msk_path = input("Enter directory path to the mask files (default: Ressources/masks): ") or args.msk_path
            elif change == 3:
                args.seed = int(input("Enter seed for random number generator (default: 42): ") or args.seed )
            elif change == 4:
                args.img_aug_path = input("Enter directory path for saving augmented images (default: Ressources/img_augmented_path): ") or args.img_aug_path
            elif change == 5:
                args.msk_aug_path = input("Enter directory path for saving augmented masks (default: Ressources/msk_augmented_path): ") or args.msk_aug_path
            elif change == 6:
                args.save_in_tiff = input("save in which format,default in the same format as orignal (default: False): ") or args.save_in_tiff 
            elif change == 7:
                overwrite = bool(int(input("do you want add new operations (0) or overwrite all operations (1), default value 0: ") or 0))
                if overwrite:
                    args.processes = input("Enter list of numbers to apply augmentation and separated numbers with + (choices: 1:rotation, 2:h_flip, 3:v_flip, 4:h_transl, 5:v_transl): ") or args.processes
                else: 
                    new_processes = input("Enter list of numbers to apply augmentation and separated numbers with + (choices: 1:rotation, 2:h_flip, 3:v_flip, 4:h_transl, 5:v_transl): ") or None
                    if new_processes != '' and new_processes is not None: args.processes = args.processes + "+" + new_processes
                    if args.processes != '' and args.processes is not None: 
                        processes = [num2op[num] for num in args.processes.split('+')]
            elif change == 8:
                execute = bool(int(input("do you really want to run augumentator with current setting? 1 for yes and 0 for no: ") or 0))
                if execute:
                    augmenter = Augumentator(args.img_path, args.msk_path, args.seed)
                    augmenter.global_inplace(True)
                    process(args)
                    augmenter.save(img_path=args.img_aug_path,mask_path=args.msk_aug_path,save_in_tiff=args.save_in_tiff)
            elif change == 9:
                to_exit = 1
            elif change == 0:
                print("no input, enter 8 to exit if you want stop")
  
                    
        

if __name__ == "__main__":
    main()