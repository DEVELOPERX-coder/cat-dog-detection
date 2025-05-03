# Cat and Dog Detector in Pure C++

## Overview

This project implements a machine learning classifier to detect whether images contain cats or dogs, built entirely in C++ with minimal external dependencies. Instead of using complex deep learning libraries, it uses traditional computer vision techniques and a simple logistic regression classifier, making it an excellent educational project for understanding image classification fundamentals.

## Features

- **Pure C++ implementation** with minimal external dependencies
- **JPEG/JPG image support** using the lightweight STB library
- **Feature extraction** using multiple techniques:
  - HOG (Histogram of Oriented Gradients) for shape detection
  - Edge features using Sobel operators
  - Histogram features for color/intensity distribution
- **Logistic regression classifier** trained with stochastic gradient descent
- **Command-line interface** for training, testing, and classifying images

## Prerequisites

- **C++17 compatible compiler**:
  - GCC 7+ (Linux)
  - MSVC 2017+ (Windows)
  - Clang 5+ (macOS)
- **CMake** (version 3.10 or higher)
- **Image dataset** of cats and dogs (JPEG format)

## Project Structure

```
cat-dog-detector/
├── include/                   # Header-only libraries
│   ├── stb_image.h            # Image loading library
│   └── stb_image_write.h      # Image writing library
├── src/
│   ├── image_processor.h      # Image loading and processing
│   ├── image_processor.cpp
│   ├── feature_extractor.h    # Feature extraction
│   ├── feature_extractor.cpp
│   ├── classifier.h           # Classification
│   ├── classifier.cpp
│   ├── utils.h                # Utility functions
│   └── main.cpp               # Main application
├── data/                      # Dataset directory
│   ├── training/
│   │   ├── cats/              # Training images of cats
│   │   └── dogs/              # Training images of dogs
│   └── testing/
│       ├── cats/              # Testing images of cats
│       └── dogs/              # Testing images of dogs
├── CMakeLists.txt             # Build configuration
└── README.md                  # This file
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/cat-dog-detector.git
cd cat-dog-detector
```

### 2. Download STB Libraries

Download these single-header libraries and place them in the `include` directory:

- [stb_image.h](https://raw.githubusercontent.com/nothings/stb/master/stb_image.h)
- [stb_image_write.h](https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h)

You can download them manually from a browser, or use the following commands:

#### Windows (PowerShell):

```powershell
mkdir -p include
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/nothings/stb/master/stb_image.h" -OutFile "include\stb_image.h"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h" -OutFile "include\stb_image_write.h"
```

#### Linux/macOS:

```bash
mkdir -p include
curl -o include/stb_image.h https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
curl -o include/stb_image_write.h https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h
```

### 3. Prepare the Dataset

#### Option A: Use the Kaggle Dogs vs. Cats Dataset

1. Download the dataset from [Kaggle Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data)
2. Extract the ZIP file
3. Create the data directories:

```bash
mkdir -p data/training/cats data/training/dogs data/testing/cats data/testing/dogs
```

4. Distribute the images:

#### Windows (PowerShell):

```powershell
# Navigate to the extracted dataset directory
cd path\to\extracted\dataset\train

# Create separate folders for cats and dogs
mkdir -p cats dogs

# Move cat and dog images to respective folders
Get-ChildItem -Filter "cat.*" | Move-Item -Destination cats\
Get-ChildItem -Filter "dog.*" | Move-Item -Destination dogs\

# Sample commands to move images to training/testing directories
# Adjust paths as needed
$catFiles = Get-ChildItem -Path cats\*.jpg
$dogFiles = Get-ChildItem -Path dogs\*.jpg

# Move first 1000 cat images to training
$catFiles | Select-Object -First 1000 | ForEach-Object {
    Copy-Item $_.FullName -Destination "C:\path\to\cat-dog-detector\data\training\cats\"
}

# Move next 250 cat images to testing
$catFiles | Select-Object -Skip 1000 -First 250 | ForEach-Object {
    Copy-Item $_.FullName -Destination "C:\path\to\cat-dog-detector\data\testing\cats\"
}

# Move first 1000 dog images to training
$dogFiles | Select-Object -First 1000 | ForEach-Object {
    Copy-Item $_.FullName -Destination "C:\path\to\cat-dog-detector\data\training\dogs\"
}

# Move next 250 dog images to testing
$dogFiles | Select-Object -Skip 1000 -First 250 | ForEach-Object {
    Copy-Item $_.FullName -Destination "C:\path\to\cat-dog-detector\data\testing\dogs\"
}
```

#### Linux/macOS:

```bash
# Navigate to the extracted dataset directory
cd path/to/extracted/dataset/train

# Create separate folders for cats and dogs
mkdir -p cats dogs

# Move cat and dog images to respective folders
mv cat.*.jpg cats/
mv dog.*.jpg dogs/

# Move first 1000 cat images to training
ls cats/*.jpg | head -1000 | xargs -I {} cp {} /path/to/cat-dog-detector/data/training/cats/

# Move next 250 cat images to testing
ls cats/*.jpg | tail -n +1001 | head -250 | xargs -I {} cp {} /path/to/cat-dog-detector/data/testing/cats/

# Move first 1000 dog images to training
ls dogs/*.jpg | head -1000 | xargs -I {} cp {} /path/to/cat-dog-detector/data/training/dogs/

# Move next 250 dog images to testing
ls dogs/*.jpg | tail -n +1001 | head -250 | xargs -I {} cp {} /path/to/cat-dog-detector/data/testing/dogs/
```

#### Option B: Use Any Cat and Dog Images

You can also use your own collection of cat and dog images:

1. Place cat images in `data/training/cats/` and `data/testing/cats/`
2. Place dog images in `data/training/dogs/` and `data/testing/dogs/`
3. Use approximately 80% of images for training and 20% for testing

## Building the Project

### Windows

With MinGW-w64:

```bash
mkdir build
cd build
cmake -G "MinGW Makefiles" ..
mingw32-make
```

With Visual Studio:

```bash
mkdir build
cd build
cmake -G "Visual Studio 17 2022" ..
cmake --build . --config Release
```

### Linux/macOS

```bash
mkdir build
cd build
cmake ..
make
```

## Usage

The application provides three main functions: training, testing, and classification.

### Training the Model

```bash
# Windows (MinGW)
cat_dog_detector.exe train ..\data\training\ model.bin

# Windows (Visual Studio)
.\Release\cat_dog_detector.exe train ..\data\training\ model.bin

# Linux/macOS
./cat_dog_detector train ../data/training/ model.bin
```

This will:

- Load all cat and dog images from the training directory
- Extract features from each image using HOG, edge detection, and histograms
- Train the logistic regression classifier
- Save the trained model to "model.bin"

The training process might take several minutes depending on the number of images and your computer's performance.

### Testing the Model

```bash
# Windows (MinGW)
cat_dog_detector.exe test ..\data\testing\ model.bin

# Windows (Visual Studio)
.\Release\cat_dog_detector.exe test ..\data\testing\ model.bin

# Linux/macOS
./cat_dog_detector test ../data/testing/ model.bin
```

This will:

- Load the trained model
- Process all images in the testing directory
- Report the accuracy of the model

### Classifying a Single Image

```bash
# Windows (MinGW)
cat_dog_detector.exe classify path\to\image.jpg model.bin

# Windows (Visual Studio)
.\Release\cat_dog_detector.exe classify path\to\image.jpg model.bin

# Linux/macOS
./cat_dog_detector classify path/to/image.jpg model.bin
```

This will:

- Load the trained model
- Process the specified image
- Output whether the image contains a cat or a dog

## Implementation Details

### Image Processing

The `ImageProcessor` class handles:

- Loading JPEG images using the STB library
- Resizing images to a standard size (64x64 pixels)
- Converting images to grayscale
- Normalizing pixel values to the range [0, 1]

### Feature Extraction

The `FeatureExtractor` class implements multiple feature extraction techniques:

1. **HOG (Histogram of Oriented Gradients)**:

   - Captures shape information by analyzing gradients
   - Divides the image into cells and computes histograms of gradient orientations
   - Normalizes histograms for better feature invariance

2. **Edge Features**:

   - Detects edges using Sobel operators
   - Creates a histogram of edge orientations
   - Helps identify boundaries between objects

3. **Histogram Features**:
   - Analyzes the distribution of pixel intensities
   - Captures global image characteristics

### Classification

The `Classifier` class implements a binary logistic regression classifier:

- Trained using stochastic gradient descent (SGD)
- Uses a sigmoid function to produce probability estimates
- Classifies images as CAT or DOG based on a probability threshold
- Includes model saving and loading functionality

## Performance Expectations

Since this is a simple machine learning implementation without deep learning:

- Expect accuracy around 65-80% depending on your dataset
- Training time: a few minutes for a thousand images
- Classification time: less than a second per image

## Troubleshooting

### Common Issues:

1. **"No images found"**:

   - Make sure your dataset directories are correctly structured
   - Verify that image files are in JPEG format (.jpg or .jpeg)

2. **Build errors**:

   - Ensure all source files contain the correct code
   - Verify that STB header files are in the include directory
   - Make sure you're using a C++17 compatible compiler

3. **Memory issues during training**:

   - The program loads all training images into memory
   - If you encounter out-of-memory errors, reduce the number of training images

4. **Low accuracy**:

   - This is a simple implementation; lower accuracy than deep learning methods is expected
   - Try increasing the training set size
   - Experiment with different feature extraction parameters

5. **Command not found**:
   - Make sure you're in the correct directory
   - On Windows, don't forget to add .exe to the command

## Contributing

Contributions are welcome! Here are some ideas for improvements:

- Add support for more image formats
- Implement more feature extraction techniques
- Add more sophisticated classification algorithms
- Create a simple GUI

## License

This project is released under the MIT License.

## Acknowledgments

- [STB](https://github.com/nothings/stb) - Single-file public domain libraries for C/C++
- [Kaggle Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats) - Dataset for training and testing
