#include "image_processor.h"
#include <iostream>
#include <cmath>

// STB image implementation
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

ImageProcessor::ImageProcessor() : data(nullptr), width(0), height(0), channels(0)
{
}

ImageProcessor::~ImageProcessor()
{
}

bool ImageProcessor::loadImage(const std::string &filename)
{
    // Free previous image if any
    data.reset();

    // Load image
    int w, h, c;
    unsigned char *rawData = stbi_load(filename.c_str(), &w, &h, &c, 0);

    if (!rawData)
    {
        std::cerr << "Error loading image: " << filename << " - "
                  << stbi_failure_reason() << std::endl;
        return false;
    }

    // Store image data
    width = w;
    height = h;
    channels = c;
    data.reset(rawData);

    std::cout << "Loaded image: " << filename << " (" << width << "x" << height
              << ", " << channels << " channels)" << std::endl;
    return true;
}

bool ImageProcessor::resize(int newWidth, int newHeight)
{
    if (!data)
    {
        std::cerr << "No image loaded" << std::endl;
        return false;
    }

    // Allocate memory for resized image
    auto resizedData = std::make_unique<unsigned char[]>(newWidth * newHeight * channels);

    // Simple nearest neighbor resize
    for (int y = 0; y < newHeight; y++)
    {
        for (int x = 0; x < newWidth; x++)
        {
            // Map to original coordinates
            int srcX = static_cast<int>(x * width / static_cast<float>(newWidth));
            int srcY = static_cast<int>(y * height / static_cast<float>(newHeight));

            // Clamp values to valid range
            srcX = std::min(std::max(srcX, 0), width - 1);
            srcY = std::min(std::max(srcY, 0), height - 1);

            // Copy pixel values
            for (int c = 0; c < channels; c++)
            {
                resizedData[(y * newWidth + x) * channels + c] =
                    data[(srcY * width + srcX) * channels + c];
            }
        }
    }

    // Update image data
    data = std::move(resizedData);
    width = newWidth;
    height = newHeight;

    return true;
}

std::vector<float> ImageProcessor::toGrayscale() const
{
    if (!data)
    {
        return {};
    }

    std::vector<float> grayscale(width * height);

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int pixelIndex = (y * width + x) * channels;
            float gray = 0.0f;

            if (channels >= 3)
            {
                // Standard RGB to grayscale conversion
                gray = 0.299f * data[pixelIndex] +
                       0.587f * data[pixelIndex + 1] +
                       0.114f * data[pixelIndex + 2];
            }
            else
            {
                // If it's already grayscale
                gray = data[pixelIndex];
            }

            // Normalize to 0.0-1.0
            grayscale[y * width + x] = gray / 255.0f;
        }
    }

    return grayscale;
}

std::vector<float> ImageProcessor::getLinearizedData() const
{
    return toGrayscale();
}

bool ImageProcessor::saveImage(const std::string &filename) const
{
    if (!data)
    {
        std::cerr << "No image to save" << std::endl;
        return false;
    }

    // Check filename extension
    size_t dotPos = filename.find_last_of('.');
    if (dotPos == std::string::npos)
    {
        std::cerr << "Filename has no extension" << std::endl;
        return false;
    }

    std::string ext = filename.substr(dotPos + 1);

    // Save according to extension
    if (ext == "png")
    {
        return stbi_write_png(filename.c_str(), width, height, channels, data.get(), width * channels) != 0;
    }
    else if (ext == "jpg" || ext == "jpeg")
    {
        return stbi_write_jpg(filename.c_str(), width, height, channels, data.get(), 95) != 0;
    }
    else if (ext == "bmp")
    {
        return stbi_write_bmp(filename.c_str(), width, height, channels, data.get()) != 0;
    }
    else
    {
        std::cerr << "Unsupported image format: " << ext << std::endl;
        return false;
    }
}