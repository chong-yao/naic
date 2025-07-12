# Poster Design Plan

## 1. Content Extraction and Summarization

### Main Title:
NAIC (AI Technical) team: CantByteUs - Kuih Classification and Segmentation

### Key Sections:

#### Introduction/Team:
- Team: CantByteUs (NAIC AI Technical Team)
- Members: Chin Zhi Xian, Ng Tze Yang, Ong Chong Yao, Terrence Ong Jun Han

#### Data Collection & Preparation:
- Scraped ~2500 images/class from Bing/Google.
- Removed duplicates by comparing tensor differences.
- Manually filtered unrelated images.
- Annotated original dataset for segmentation using Label Studio.
- Included varied images: high/low-res, varied lighting/angles, partially eaten kuih.
- Pseudo Labelling: Semi-supervised annotation method.
  - Annotated 20-30 kuih/class based on feature complexity.
  - Trained small YOLOv11-seg for faster annotation, then manual verification.
  - Retrained larger model with combined data.
- Final raw dataset: 98 images/class (perfectly annotated).
- Split: 90 images for training, 8 for validation per class.
- Roboflow augmentation: Tripled dataset size (784 images total), excluded hue/color adjustments.
- Added non-kuih images to reduce False Positives.

#### Model Development:
- Tools: Own computers, CUDA, PyTorch, Ultralytics.
- Initial Approach: Edited 'yolo11seg.yaml' (increased depth, width, channel capacity), added attention layers.
- Problem: Large models overfit on small datasets.
- Solution: Pre-trained YOLOv11x-seg on COCO 2017, then fine-tuned on kuih dataset.
- Preprocessed test images for consistent light balance.

#### Final Model: Ensemble Approach (CNN Segmentation + Vision Transformer)
- **Why Segmentation?**
  - Robust segmentation improves classification accuracy by focusing on the core object.
  - YOLOv11x-seg confusion matrix was near-perfect.
- **Why Vision Transformer (ViT)?**
  - Splits image into patches, transforms to tokens (like LLMs).
  - Captures relationships across entire image in every layer.
  - Addresses visual similarities (e.g., Kek Lapis vs. Kuih Lapis) by analyzing textures and long-distance relationships.
  - Performs well when pre-trained on large datasets (e.g., ImageNet 22k).
  - Fast convergence, but careful saving needed to prevent overfitting.

#### Final Output: Combining CNN and ViT
- Soft voting method for combined output.
- Agreement: Chosen class.
- Disagreement: Class with highest confidence from either model.
- Hybrid approach outperformed solo models.

## 2. Visual Hierarchy & Content Organization

### A2 Size (420 x 594 mm or 16.5 x 23.4 inches)

### Layout:
- **Top Section (Most Important):** Main Title, Team Members, and a concise problem statement/project goal.
- **Middle Section:** Data Collection & Preparation (Problem/Solution flow), Model Development (Initial Approach & Solution).
- **Bottom Section:** Final Model (Ensemble Approach - Why CNN, Why ViT), Final Output (Combining CNN & ViT).
- **Margins:** At least 1 inch (25.4 mm) on all sides.
- **Spacing:** Consistent spacing between sections and elements.
- **Grouping:** Use boxes or background colors to group related information (e.g., Data Collection, Model Development, Ensemble Approach).

### Reading Path:
- Clear flow from top to bottom, left to right within sections.
- Use larger font sizes and bolding for headers.
- Use bullet points for key information.

## 3. Readability

### Font Families (Max 2):
- **Header Font:** A clear, bold sans-serif font (e.g., Montserrat, Open Sans Bold).
- **Body Font:** A readable sans-serif font (e.g., Open Sans, Lato, Roboto).

### Font Sizes:
- **Main Title:** Very large (e.g., 72pt+)
- **Section Headers:** Large (e.g., 48-60pt)
- **Sub-headers:** Medium (e.g., 36-42pt)
- **Body Text/Bullet Points:** Readable (e.g., 24-30pt)
- **Smallest Text (e.g., image captions):** 18-20pt (ensure readability)

### Color Contrast:
- High contrast between text and background. Avoid light yellow text.
- Example: Dark text on light background, or light text on dark background.

## 4. Visual Elements

### Color Palette (2-3 main colors + black/white):
- Suggestion: A vibrant, food-related color (e.g., a warm orange/yellow or a rich green/blue) as accent, combined with a neutral primary color (e.g., a muted grey or cream) and black/white for text/background.
- Example: Primary: #F5F5DC (Cream), Accent 1: #FF6B6B (Coral Red), Accent 2: #4ECDC4 (Turquoise), Text: #333333 (Dark Grey), Background: #FFFFFF (White).

### Images/Graphics:
- **High-quality images:** Crucial for large prints. Will search for relevant images of kuih, AI/ML concepts, or abstract representations.
- **Charts/Diagrams:** The README mentions confusion matrices and training metrics. If possible, recreate or simplify these visually. A simple diagram explaining pseudo-labelling or the ensemble approach would be beneficial.
- **Purposeful visuals:** Every visual must serve to explain or enhance the content.

### Specific Visual Ideas:
- **Hero Image:** A high-quality, appealing image of various kuih at the top.
- **Data Collection:** Icons representing data scraping, filtering, annotation.
- **Pseudo-labelling:** A simplified flow diagram.
- **Model Development:** Icons for PyTorch, CUDA, etc.
- **Ensemble Model:** A diagram showing CNN and ViT inputs combining for a final output.
- **Confusion Matrix/Metrics:** Simplified visual representation if possible, or a note about their effectiveness.

## 5. Software/Tools for Creation
- Will use web technologies (HTML/CSS/JS) for precise layout and responsive design, then convert to PDF for A2 printing. This allows for programmatic control over elements and ensures high quality for large prints.

