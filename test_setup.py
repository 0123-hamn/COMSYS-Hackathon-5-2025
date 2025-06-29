import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def test_data_structure():
    """Test if the data structure is correct"""
    print("Testing data structure...")
    
    # Check if directories exist
    train_dir = 'train'
    val_dir = 'val'
    
    if not os.path.exists(train_dir):
        print(f"❌ Training directory '{train_dir}' not found!")
        return False
    
    if not os.path.exists(val_dir):
        print(f"❌ Validation directory '{val_dir}' not found!")
        return False
    
    # Check subdirectories
    train_female = os.path.join(train_dir, 'female')
    train_male = os.path.join(train_dir, 'male')
    val_female = os.path.join(val_dir, 'female')
    val_male = os.path.join(val_dir, 'male')
    
    directories = [train_female, train_male, val_female, val_male]
    
    for dir_path in directories:
        if not os.path.exists(dir_path):
            print(f"❌ Directory '{dir_path}' not found!")
            return False
    
    print("✅ All directories found!")
    return True

def count_images():
    """Count images in each directory"""
    print("\nCounting images...")
    
    directories = {
        'train/female': 'train/female',
        'train/male': 'train/male',
        'val/female': 'val/female',
        'val/male': 'val/male'
    }
    
    total_images = 0
    
    for name, path in directories.items():
        if os.path.exists(path):
            # Count image files
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_count = 0
            
            for file in os.listdir(path):
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_count += 1
            
            print(f"📁 {name}: {image_count} images")
            total_images += image_count
        else:
            print(f"❌ {name}: Directory not found")
    
    print(f"\n📊 Total images: {total_images}")
    return total_images

def test_image_loading():
    """Test if images can be loaded properly"""
    print("\nTesting image loading...")
    
    # Try to load a few images from each directory
    directories = ['train/female', 'train/male', 'val/female', 'val/male']
    
    for dir_path in directories:
        if os.path.exists(dir_path):
            files = [f for f in os.listdir(dir_path) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            if files:
                # Try to load the first image
                test_image_path = os.path.join(dir_path, files[0])
                try:
                    # Test with PIL
                    img_pil = Image.open(test_image_path)
                    img_array = np.array(img_pil)
                    
                    # Test with OpenCV
                    img_cv = cv2.imread(test_image_path)
                    
                    print(f"✅ {dir_path}: Successfully loaded {files[0]} "
                          f"({img_array.shape[1]}x{img_array.shape[0]})")
                    
                except Exception as e:
                    print(f"❌ {dir_path}: Failed to load {files[0]} - {str(e)}")
            else:
                print(f"⚠️  {dir_path}: No images found")

def test_webcam():
    """Test if webcam is available"""
    print("\nTesting webcam...")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Webcam not available or already in use")
        return False
    
    # Try to read a frame
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        print(f"✅ Webcam available - Resolution: {frame.shape[1]}x{frame.shape[0]}")
        return True
    else:
        print("❌ Could not read frame from webcam")
        return False

def display_sample_images():
    """Display sample images from the dataset"""
    print("\nDisplaying sample images...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Sample Images from Dataset', fontsize=16)
    
    directories = ['train/female', 'train/male', 'val/female', 'val/male']
    titles = ['Train Female', 'Train Male', 'Val Female', 'Val Male']
    
    for i, (dir_path, title) in enumerate(zip(directories, titles)):
        if os.path.exists(dir_path):
            files = [f for f in os.listdir(dir_path) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            if files:
                # Load and display first image
                img_path = os.path.join(dir_path, files[0])
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                row, col = i // 2, i % 2
                axes[row, col].imshow(img_rgb)
                axes[row, col].set_title(f"{title}\n{files[0]}")
                axes[row, col].axis('off')
            else:
                axes[i // 2, i % 2].text(0.5, 0.5, f"No images in {dir_path}", 
                                       ha='center', va='center', transform=axes[i // 2, i % 2].transAxes)
                axes[i // 2, i % 2].set_title(title)
                axes[i // 2, i % 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_images.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Run all tests"""
    print("🔍 Gender Recognition System - Setup Test")
    print("=" * 50)
    
    # Test data structure
    if not test_data_structure():
        print("\n❌ Data structure test failed!")
        return
    
    # Count images
    total_images = count_images()
    
    if total_images == 0:
        print("\n❌ No images found in the dataset!")
        return
    
    # Test image loading
    test_image_loading()
    
    # Test webcam
    webcam_available = test_webcam()
    
    # Display sample images
    try:
        display_sample_images()
    except Exception as e:
        print(f"⚠️  Could not display sample images: {str(e)}")
    
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print(f"✅ Data structure: OK")
    print(f"✅ Total images: {total_images}")
    print(f"✅ Image loading: OK")
    print(f"{'✅' if webcam_available else '❌'} Webcam: {'Available' if webcam_available else 'Not available'}")
    
    if total_images > 0:
        print(f"\n🎉 Setup test completed successfully!")
        print(f"📝 You can now run: python train_gender_model.py")
        
        if webcam_available:
            print(f"📹 After training, you can run: python webcam_gender_recognition.py")
        else:
            print(f"⚠️  Webcam not available - you can still train the model but won't be able to use webcam features")
    else:
        print(f"\n❌ Setup test failed - no images found!")

if __name__ == "__main__":
    main() 