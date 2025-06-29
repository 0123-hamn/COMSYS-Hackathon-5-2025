import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

def create_model_diagram():
    """Create a comprehensive diagram of the gender recognition model architecture"""
    
    # Create figure with larger size
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))
    fig.suptitle('Gender Recognition System Architecture', fontsize=20, fontweight='bold')
    
    # Left subplot: Model Architecture
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 12)
    ax1.set_title('Deep Learning Model Architecture', fontsize=16, fontweight='bold')
    ax1.axis('off')
    
    # Colors for different components
    colors = {
        'input': '#E8F4FD',
        'vgg16': '#FFE6E6',
        'feature_extraction': '#E6FFE6',
        'classification': '#FFF2E6',
        'output': '#F0E6FF'
    }
    
    # Input Layer
    input_box = FancyBboxPatch((1, 10), 2, 1.5, 
                              boxstyle="round,pad=0.1", 
                              facecolor=colors['input'], 
                              edgecolor='black', linewidth=2)
    ax1.add_patch(input_box)
    ax1.text(2, 10.75, 'Input Image\n224×224×3', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # VGG16 Base Model
    vgg_box = FancyBboxPatch((1, 7.5), 2, 2, 
                            boxstyle="round,pad=0.1", 
                            facecolor=colors['vgg16'], 
                            edgecolor='black', linewidth=2)
    ax1.add_patch(vgg_box)
    ax1.text(2, 8.5, 'VGG16 Base Model\n(Pre-trained on ImageNet)\n\n• Conv2D Layers\n• MaxPooling2D\n• BatchNormalization\n\nFrozen during training', 
             ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Feature Extraction
    feature_box = FancyBboxPatch((1, 5.5), 2, 1.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['feature_extraction'], 
                                edgecolor='black', linewidth=2)
    ax1.add_patch(feature_box)
    ax1.text(2, 6.25, 'Global Average\nPooling2D\n\n512 features', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Classification Layers
    dense1_box = FancyBboxPatch((1, 3.5), 2, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['classification'], 
                               edgecolor='black', linewidth=2)
    ax1.add_patch(dense1_box)
    ax1.text(2, 4, 'Dense(512)\nReLU + Dropout(0.5)', ha='center', va='center', fontsize=9, fontweight='bold')
    
    dense2_box = FancyBboxPatch((1, 2), 2, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['classification'], 
                               edgecolor='black', linewidth=2)
    ax1.add_patch(dense2_box)
    ax1.text(2, 2.5, 'Dense(256)\nReLU + Dropout(0.3)', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Output Layer
    output_box = FancyBboxPatch((1, 0.5), 2, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=colors['output'], 
                               edgecolor='black', linewidth=2)
    ax1.add_patch(output_box)
    ax1.text(2, 1, 'Dense(1)\nSigmoid\n\nMale/Female', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrows connecting layers
    arrow_props = dict(arrowstyle='->', lw=2, color='blue')
    
    # Input to VGG16
    ax1.annotate('', xy=(2, 7.5), xytext=(2, 9.5), arrowprops=arrow_props)
    
    # VGG16 to Feature Extraction
    ax1.annotate('', xy=(2, 5.5), xytext=(2, 7.5), arrowprops=arrow_props)
    
    # Feature Extraction to Dense1
    ax1.annotate('', xy=(2, 3.5), xytext=(2, 5.5), arrowprops=arrow_props)
    
    # Dense1 to Dense2
    ax1.annotate('', xy=(2, 2), xytext=(2, 3.5), arrowprops=arrow_props)
    
    # Dense2 to Output
    ax1.annotate('', xy=(2, 0.5), xytext=(2, 2), arrowprops=arrow_props)
    
    # Right subplot: System Pipeline
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 12)
    ax2.set_title('Real-time Webcam Pipeline', fontsize=16, fontweight='bold')
    ax2.axis('off')
    
    # Webcam Input
    webcam_box = FancyBboxPatch((0.5, 10.5), 2, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#E6F3FF', 
                               edgecolor='black', linewidth=2)
    ax2.add_patch(webcam_box)
    ax2.text(1.5, 11, 'Webcam Feed\n640×480', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Face Detection
    face_det_box = FancyBboxPatch((3.5, 10.5), 2, 1, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor='#FFE6E6', 
                                 edgecolor='black', linewidth=2)
    ax2.add_patch(face_det_box)
    ax2.text(4.5, 11, 'Face Detection\nHaar Cascade', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Face Extraction
    face_ext_box = FancyBboxPatch((6.5, 10.5), 2, 1, 
                                 boxstyle="round,pad=0.1", 
                                 facecolor='#E6FFE6', 
                                 edgecolor='black', linewidth=2)
    ax2.add_patch(face_ext_box)
    ax2.text(7.5, 11, 'Face Extraction\nBounding Box', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Preprocessing
    preprocess_box = FancyBboxPatch((1, 8), 3, 1, 
                                   boxstyle="round,pad=0.1", 
                                   facecolor='#FFF2E6', 
                                   edgecolor='black', linewidth=2)
    ax2.add_patch(preprocess_box)
    ax2.text(2.5, 8.5, 'Preprocessing\nResize(224×224) + Normalize + RGB', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Model Prediction
    model_box = FancyBboxPatch((5, 8), 3, 1, 
                              boxstyle="round,pad=0.1", 
                              facecolor='#F0E6FF', 
                              edgecolor='black', linewidth=2)
    ax2.add_patch(model_box)
    ax2.text(6.5, 8.5, 'Gender Classification\nTrained Model', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Prediction History
    history_box = FancyBboxPatch((1, 5.5), 3, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor='#E6F3FF', 
                                edgecolor='black', linewidth=2)
    ax2.add_patch(history_box)
    ax2.text(2.5, 6, 'Prediction History\nRolling Window (10 frames)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Stable Prediction
    stable_box = FancyBboxPatch((5, 5.5), 3, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor='#FFE6E6', 
                               edgecolor='black', linewidth=2)
    ax2.add_patch(stable_box)
    ax2.text(6.5, 6, 'Stable Prediction\nMajority Voting', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Visualization
    viz_box = FancyBboxPatch((1, 3), 7, 1.5, 
                            boxstyle="round,pad=0.1", 
                            facecolor='#E6FFE6', 
                            edgecolor='black', linewidth=2)
    ax2.add_patch(viz_box)
    ax2.text(4.5, 3.75, 'Visualization Output\n\n• Color-coded bounding boxes (Pink=Female, Blue=Male)\n• Confidence scores\n• FPS counter\n• Face count', 
             ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrows for pipeline
    # Webcam to Face Detection
    ax2.annotate('', xy=(3.5, 11), xytext=(2.5, 11), arrowprops=arrow_props)
    
    # Face Detection to Face Extraction
    ax2.annotate('', xy=(6.5, 11), xytext=(5.5, 11), arrowprops=arrow_props)
    
    # Face Extraction to Preprocessing
    ax2.annotate('', xy=(2.5, 8), xytext=(7.5, 10.5), arrowprops=arrow_props)
    
    # Preprocessing to Model
    ax2.annotate('', xy=(5, 8.5), xytext=(4, 8.5), arrowprops=arrow_props)
    
    # Model to History
    ax2.annotate('', xy=(2.5, 5.5), xytext=(6.5, 8), arrowprops=arrow_props)
    
    # History to Stable
    ax2.annotate('', xy=(5, 6), xytext=(4.5, 6), arrowprops=arrow_props)
    
    # Stable to Visualization
    ax2.annotate('', xy=(4.5, 3), xytext=(6.5, 5.5), arrowprops=arrow_props)
    
    # Add model specifications
    spec_text = """
Model Specifications:
• Input Size: 224×224×3 RGB
• Base Model: VGG16 (Transfer Learning)
• Training Data: 1,926 images (303♀ + 1,623♂)
• Validation Data: 422 images (79♀ + 343♂)
• Optimizer: Adam
• Loss: Binary Crossentropy
• Batch Size: 32
• Epochs: 50 (Early Stopping)
• Data Augmentation: Rotation, Scaling, Flipping
    """
    
    ax1.text(5, 6, spec_text, fontsize=9, bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    # Add performance features
    perf_text = """
Performance Features:
• Real-time Processing: ~30 FPS
• Multi-face Detection
• Confidence Threshold: 0.7
• Prediction Stability: 10-frame window
• Face Detection: Haar Cascade
• Screenshot Capability
• FPS Monitoring
    """
    
    ax2.text(0.5, 1, perf_text, fontsize=9, bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('model_architecture_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Model architecture diagram saved as 'model_architecture_diagram.png'")

def create_data_flow_diagram():
    """Create a data flow diagram showing the training and inference process"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    fig.suptitle('Gender Recognition System - Data Flow', fontsize=20, fontweight='bold')
    
    # Training Flow (top)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 6)
    ax1.set_title('Training Pipeline', fontsize=16, fontweight='bold')
    ax1.axis('off')
    
    # Training data boxes
    train_female = FancyBboxPatch((0.5, 4), 1.5, 1, boxstyle="round,pad=0.1", 
                                 facecolor='#FFB6C1', edgecolor='black', linewidth=2)
    ax1.add_patch(train_female)
    ax1.text(1.25, 4.5, 'Train\nFemale\n(303)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    train_male = FancyBboxPatch((2.5, 4), 1.5, 1, boxstyle="round,pad=0.1", 
                               facecolor='#87CEEB', edgecolor='black', linewidth=2)
    ax1.add_patch(train_male)
    ax1.text(3.25, 4.5, 'Train\nMale\n(1,623)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Data augmentation
    aug_box = FancyBboxPatch((5, 4), 2, 1, boxstyle="round,pad=0.1", 
                            facecolor='#FFE6E6', edgecolor='black', linewidth=2)
    ax1.add_patch(aug_box)
    ax1.text(6, 4.5, 'Data Augmentation\nRotation, Scaling, Flip', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Model training
    model_box = FancyBboxPatch((7.5, 4), 2, 1, boxstyle="round,pad=0.1", 
                              facecolor='#E6FFE6', edgecolor='black', linewidth=2)
    ax1.add_patch(model_box)
    ax1.text(8.5, 4.5, 'Model Training\nVGG16 + Dense Layers', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Validation
    val_female = FancyBboxPatch((0.5, 2), 1.5, 1, boxstyle="round,pad=0.1", 
                               facecolor='#FFB6C1', edgecolor='black', linewidth=2)
    ax1.add_patch(val_female)
    ax1.text(1.25, 2.5, 'Val\nFemale\n(79)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    val_male = FancyBboxPatch((2.5, 2), 1.5, 1, boxstyle="round,pad=0.1", 
                             facecolor='#87CEEB', edgecolor='black', linewidth=2)
    ax1.add_patch(val_male)
    ax1.text(3.25, 2.5, 'Val\nMale\n(343)', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Evaluation
    eval_box = FancyBboxPatch((5, 2), 2, 1, boxstyle="round,pad=0.1", 
                             facecolor='#FFF2E6', edgecolor='black', linewidth=2)
    ax1.add_patch(eval_box)
    ax1.text(6, 2.5, 'Model Evaluation\nAccuracy, Loss, Confusion Matrix', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Model save
    save_box = FancyBboxPatch((7.5, 2), 2, 1, boxstyle="round,pad=0.1", 
                             facecolor='#F0E6FF', edgecolor='black', linewidth=2)
    ax1.add_patch(save_box)
    ax1.text(8.5, 2.5, 'Save Model\nbest_gender_model.h5', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrows for training
    arrow_props = dict(arrowstyle='->', lw=2, color='blue')
    
    # Training data to augmentation
    ax1.annotate('', xy=(5, 4.5), xytext=(2, 4.5), arrowprops=arrow_props)
    ax1.annotate('', xy=(5, 4.5), xytext=(4, 4.5), arrowprops=arrow_props)
    
    # Augmentation to training
    ax1.annotate('', xy=(7.5, 4.5), xytext=(7, 4.5), arrowprops=arrow_props)
    
    # Validation to evaluation
    ax1.annotate('', xy=(5, 2.5), xytext=(2, 2.5), arrowprops=arrow_props)
    ax1.annotate('', xy=(5, 2.5), xytext=(4, 2.5), arrowprops=arrow_props)
    
    # Evaluation to save
    ax1.annotate('', xy=(7.5, 2.5), xytext=(7, 2.5), arrowprops=arrow_props)
    
    # Inference Flow (bottom)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 6)
    ax2.set_title('Real-time Inference Pipeline', fontsize=16, fontweight='bold')
    ax2.axis('off')
    
    # Webcam
    webcam_box = FancyBboxPatch((0.5, 4.5), 1.5, 1, boxstyle="round,pad=0.1", 
                               facecolor='#E6F3FF', edgecolor='black', linewidth=2)
    ax2.add_patch(webcam_box)
    ax2.text(1.25, 5, 'Webcam\n640×480', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Face detection
    face_box = FancyBboxPatch((2.5, 4.5), 1.5, 1, boxstyle="round,pad=0.1", 
                             facecolor='#FFE6E6', edgecolor='black', linewidth=2)
    ax2.add_patch(face_box)
    ax2.text(3.25, 5, 'Face Detection\nHaar Cascade', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Preprocessing
    preprocess_box = FancyBboxPatch((4.5, 4.5), 1.5, 1, boxstyle="round,pad=0.1", 
                                   facecolor='#E6FFE6', edgecolor='black', linewidth=2)
    ax2.add_patch(preprocess_box)
    ax2.text(5.25, 5, 'Preprocess\n224×224', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Model inference
    inference_box = FancyBboxPatch((6.5, 4.5), 1.5, 1, boxstyle="round,pad=0.1", 
                                  facecolor='#FFF2E6', edgecolor='black', linewidth=2)
    ax2.add_patch(inference_box)
    ax2.text(7.25, 5, 'Model\nInference', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Prediction
    pred_box = FancyBboxPatch((8.5, 4.5), 1.5, 1, boxstyle="round,pad=0.1", 
                             facecolor='#F0E6FF', edgecolor='black', linewidth=2)
    ax2.add_patch(pred_box)
    ax2.text(9.25, 5, 'Gender\nPrediction', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Display
    display_box = FancyBboxPatch((3, 2), 4, 1.5, boxstyle="round,pad=0.1", 
                                facecolor='#E6FFE6', edgecolor='black', linewidth=2)
    ax2.add_patch(display_box)
    ax2.text(5, 2.75, 'Real-time Display\n\n• Color-coded bounding boxes\n• Confidence scores\n• FPS counter\n• Face count', 
             ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Arrows for inference
    # Webcam to face detection
    ax2.annotate('', xy=(2.5, 5), xytext=(2, 5), arrowprops=arrow_props)
    
    # Face detection to preprocessing
    ax2.annotate('', xy=(4.5, 5), xytext=(4, 5), arrowprops=arrow_props)
    
    # Preprocessing to inference
    ax2.annotate('', xy=(6.5, 5), xytext=(6, 5), arrowprops=arrow_props)
    
    # Inference to prediction
    ax2.annotate('', xy=(8.5, 5), xytext=(8, 5), arrowprops=arrow_props)
    
    # Prediction to display
    ax2.annotate('', xy=(5, 2), xytext=(9.25, 4.5), arrowprops=arrow_props)
    
    plt.tight_layout()
    plt.savefig('data_flow_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Data flow diagram saved as 'data_flow_diagram.png'")

if __name__ == "__main__":
    print("🎨 Creating Gender Recognition System Diagrams...")
    create_model_diagram()
    create_data_flow_diagram()
    print("\n🎉 All diagrams created successfully!")
    print("📁 Files generated:")
    print("   • model_architecture_diagram.png")
    print("   • data_flow_diagram.png") 