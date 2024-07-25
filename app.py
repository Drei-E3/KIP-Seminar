import runpy

def run_script(script):
    runpy.run_path(script, run_name="__main__")

def main_menu():
    print("\nImage Processing and Modeling Application")
    print("1. Image Augmentation")
    print("2. Image Cropping")
    print("3. Image Grayscaling")
    print("4. Model Operations")
    print("5. Exit")
    choice = input("Enter your choice (1-5): ")
    return choice


def augmentation():
    run_script("utils/augumentation.py")

def cropping():
    run_script("utils/cropping.py")

def grayscaling():
    run_script("utils/grayscaling.py")

def modeling():
    run_script("utils/modeling.py")

if __name__ == "__main__":
    while True:
        choice = main_menu()
        if choice == '1':
            augmentation()
        elif choice == '2':
            cropping()
        elif choice == '3':
            grayscaling()
        elif choice == '4':
            modeling()
        elif choice == '5':
            print("Exiting application.")
            break
        else:
            print("Invalid choice. Please select a valid option.")