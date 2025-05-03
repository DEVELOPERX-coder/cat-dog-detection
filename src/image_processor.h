#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <string>
#include <vector>
#include <memory>

class ImageProcessor
{
private:
    std::unique_ptr<unsigned char[]> data;
    int width;
    int height;
    int channels;

public:
    ImageProcessor();
    ~ImageProcessor();

    // Load an image from file (supports JPG, PNG, etc.)
    bool loadImage(const std::string &filename);

    // Resize image to standard dimensions
    bool resize(int newWidth, int newHeight);

    // Convert to grayscale and normalize to values between 0.0 and 1.0
    std::vector<float> toGrayscale() const;

    // Get linearized pixel data
    std::vector<float> getLinearizedData() const;

    // Save processed image (for debugging)
    bool saveImage(const std::string &filename) const;

    // Getters
    int getWidth() const { return width; }
    int getHeight() const { return height; }
    int getChannels() const { return channels; }
};

#endif // IMAGE_PROCESSOR_H