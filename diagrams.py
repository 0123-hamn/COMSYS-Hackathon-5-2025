import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

def create_model_diagram():
    """Create a diagram of the gender recognition model architecture"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
    fig.suptitle('Gender Recognition System Architecture', fontsize=18, fontweight='bold')
    
    # Left: Model Architecture
    ax1.set_xlim(0, 8)
    ax1.set_ylim(0, 10)
    ax1.set_title('Deep Learning Model', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Model layers
    layers = [
        {'name': 'Input\n224×224×3', 'y': 8.5, 'color': '#E8F4FD'},
        {'name': 'VGG16 Base\n(Pre-trained)', 'y': 7, 'color': '#FFE6E6'},
        {'name': 'Global Average\nPooling', 'y': 5.5, 'color': '#E6FFE6'},
        {'name': 'Dense(512)\nReLU + Dropout', 'y': 4, 'color': '#FFF2E6'},
        {'name': 'Dense(256)\nReLU + Dropout', 'y': 2.5, 'color': '#FFF2E6'},
        {'name': 'Dense(1)\nSigmoid', 'y': 1, 'color': '#F0E6FF'}
    ]
    
    for layer in layers:
        box = FancyBboxPatch((2, layer['y']-0.5), 4, 1, 
                           boxstyle="round,pad=0.1", 
                           facecolor=layer['color'], 
                           edgecolor='black', linewidth=2)
        ax1.add_patch(box)
        ax1.text(4, layer['y'], layer['name'], ha='center', va='center', 
                fontsize=10, fontweight='bold')
    
    # Arrows
    for i in range(len(layers)-1):
        ax1.annotate('', xy=(4, layers[i+1]['y']+0.5), xytext=(4, layers[i]['y']-0.5), 
                    arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    
    # Right: System Pipeline
    ax2.set_xlim(0, 8)
    ax2.set_ylim(0, 10)
    ax2.set_title('Real-time Pipeline', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Pipeline steps
    steps = [
        {'name': 'Webcam\n640×480', 'x': 1, 'y': 8.5, 'color': '#E6F3FF'},
        {'name': 'Face Detection\nHaar Cascade', 'x': 3, 'y': 8.5, 'color': '#FFE6E6'},
        {'name': 'Face Extraction', 'x': 5, 'y': 8.5, 'color': '#E6FFE6'},
        {'name': 'Preprocessing\n224×224', 'x': 2, 'y': 6.5, 'color': '#FFF2E6'},
        {'name': 'Model Inference', 'x': 4, 'y': 6.5, 'color': '#F0E6FF'},
        {'name': 'Gender Prediction', 'x': 6, 'y': 6.5, 'color': '#F0E6FF'},
        {'name': 'Visualization\nColor-coded boxes', 'x': 3, 'y': 4, 'color': '#E6FFE6'}
    ]
    
    for step in steps:
        box = FancyBboxPatch((step['x']-0.8, step['y']-0.4), 1.6, 0.8, 
                           boxstyle="round,pad=0.1", 
                           facecolor=step['color'], 
                           edgecolor='black', linewidth=2)
        ax2.add_patch(box)
        ax2.text(step['x'], step['y'], step['name'], ha='center', va='center', 
                fontsize=9, fontweight='bold')
    
    # Pipeline arrows
    arrows = [
        ((1.8, 8.5), (2.2, 8.5)),  # Webcam to Face Detection
        ((3.8, 8.5), (4.2, 8.5)),  # Face Detection to Extraction
        ((5, 8.1), (2.8, 6.9)),    # Extraction to Preprocessing
        ((2.8, 6.5), (3.2, 6.5)),  # Preprocessing to Inference
        ((4.8, 6.5), (5.2, 6.5)),  # Inference to Prediction
        ((6, 6.1), (3.8, 4.4))     # Prediction to Visualization
    ]
    
    for start, end in arrows:
        ax2.annotate('', xy=end, xytext=start, 
                    arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    
    # Add specifications
    spec_text = """
Model Specs:
• Input: 224×224×3 RGB
• Base: VGG16 (Transfer Learning)
• Training: 1,926 images
• Validation: 422 images
• Optimizer: Adam
• Loss: Binary Crossentropy
• Batch Size: 32
• Epochs: 50 (Early Stopping)
    """
    
    ax1.text(0.5, 0.5, spec_text, fontsize=8, 
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    perf_text = """
Performance:
• Real-time: ~30 FPS
• Multi-face Detection
• Confidence Threshold: 0.7
• Prediction Stability: 10-frame window
• Face Detection: Haar Cascade
• Screenshot Capability
    """
    
    ax2.text(0.5, 0.5, perf_text, fontsize=8, 
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('model_architecture_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Model architecture diagram saved as 'model_architecture_diagram.png'")

def create_data_flow_diagram():
    """Create a data flow diagram"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Gender Recognition System - Data Flow', fontsize=18, fontweight='bold')
    
    # Training Flow
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 5)
    ax1.set_title('Training Pipeline', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Training boxes
    boxes = [
        {'name': 'Train Female\n(303)', 'x': 1, 'y': 3.5, 'color': '#FFB6C1'},
        {'name': 'Train Male\n(1,623)', 'x': 3, 'y': 3.5, 'color': '#87CEEB'},
        {'name': 'Data Augmentation\nRotation, Scaling, Flip', 'x': 6, 'y': 3.5, 'color': '#FFE6E6'},
        {'name': 'Model Training\nVGG16 + Dense', 'x': 8, 'y': 3.5, 'color': '#E6FFE6'},
        {'name': 'Val Female\n(79)', 'x': 1, 'y': 1.5, 'color': '#FFB6C1'},
        {'name': 'Val Male\n(343)', 'x': 3, 'y': 1.5, 'color': '#87CEEB'},
        {'name': 'Model Evaluation\nAccuracy, Loss', 'x': 6, 'y': 1.5, 'color': '#FFF2E6'},
        {'name': 'Save Model\nbest_gender_model.h5', 'x': 8, 'y': 1.5, 'color': '#F0E6FF'}
    ]
    
    for box in boxes:
        rect = FancyBboxPatch((box['x']-0.8, box['y']-0.4), 1.6, 0.8, 
                            boxstyle="round,pad=0.1", 
                            facecolor=box['color'], 
                            edgecolor='black', linewidth=2)
        ax1.add_patch(rect)
        ax1.text(box['x'], box['y'], box['name'], ha='center', va='center', 
                fontsize=9, fontweight='bold')
    
    # Training arrows
    arrows = [
        ((1.8, 3.5), (5.2, 3.5)),  # Train Female to Augmentation
        ((3.8, 3.5), (5.2, 3.5)),  # Train Male to Augmentation
        ((6.8, 3.5), (7.2, 3.5)),  # Augmentation to Training
        ((1.8, 1.5), (5.2, 1.5)),  # Val Female to Evaluation
        ((3.8, 1.5), (5.2, 1.5)),  # Val Male to Evaluation
        ((6.8, 1.5), (7.2, 1.5))   # Evaluation to Save
    ]
    
    for start, end in arrows:
        ax1.annotate('', xy=end, xytext=start, 
                    arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    
    # Inference Flow
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 5)
    ax2.set_title('Real-time Inference Pipeline', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Inference boxes
    inf_boxes = [
        {'name': 'Webcam\n640×480', 'x': 1, 'y': 3.5, 'color': '#E6F3FF'},
        {'name': 'Face Detection\nHaar Cascade', 'x': 3, 'y': 3.5, 'color': '#FFE6E6'},
        {'name': 'Preprocess\n224×224', 'x': 5, 'y': 3.5, 'color': '#E6FFE6'},
        {'name': 'Model\nInference', 'x': 7, 'y': 3.5, 'color': '#FFF2E6'},
        {'name': 'Gender\nPrediction', 'x': 9, 'y': 3.5, 'color': '#F0E6FF'},
        {'name': 'Real-time Display\nColor-coded boxes\nConfidence scores\nFPS counter', 'x': 5, 'y': 1.5, 'color': '#E6FFE6'}
    ]
    
    for box in inf_boxes:
        rect = FancyBboxPatch((box['x']-0.8, box['y']-0.4), 1.6, 0.8, 
                            boxstyle="round,pad=0.1", 
                            facecolor=box['color'], 
                            edgecolor='black', linewidth=2)
        ax2.add_patch(rect)
        ax2.text(box['x'], box['y'], box['name'], ha='center', va='center', 
                fontsize=9, fontweight='bold')
    
    # Inference arrows
    inf_arrows = [
        ((1.8, 3.5), (2.2, 3.5)),  # Webcam to Face Detection
        ((3.8, 3.5), (4.2, 3.5)),  # Face Detection to Preprocess
        ((5.8, 3.5), (6.2, 3.5)),  # Preprocess to Inference
        ((7.8, 3.5), (8.2, 3.5)),  # Inference to Prediction
        ((9, 3.1), (5.8, 1.9))     # Prediction to Display
    ]
    
    for start, end in inf_arrows:
        ax2.annotate('', xy=end, xytext=start, 
                    arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    
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