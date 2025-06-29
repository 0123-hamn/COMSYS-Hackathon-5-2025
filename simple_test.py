import os

def test_data_structure():
    """Test if the data structure is correct"""
    print("🔍 Testing Gender Recognition System Setup")
    print("=" * 50)
    
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
    print("\n📊 Counting images...")
    
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

def check_file_structure():
    """Check if all required files are present"""
    print("\n📁 Checking file structure...")
    
    required_files = [
        'train_gender_model.py',
        'webcam_gender_recognition.py',
        'requirements.txt',
        'README.md'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
            missing_files.append(file)
    
    return len(missing_files) == 0

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
    
    # Check file structure
    files_ok = check_file_structure()
    
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print(f"✅ Data structure: OK")
    print(f"✅ Total images: {total_images}")
    print(f"{'✅' if files_ok else '❌'} Required files: {'All present' if files_ok else 'Some missing'}")
    
    if total_images > 0:
        print(f"\n🎉 Setup test completed successfully!")
        print(f"📝 Next steps:")
        print(f"   1. Install dependencies: pip install -r requirements.txt")
        print(f"   2. Train the model: python train_gender_model.py")
        print(f"   3. Run webcam app: python webcam_gender_recognition.py")
    else:
        print(f"\n❌ Setup test failed - no images found!")

if __name__ == "__main__":
    main() 