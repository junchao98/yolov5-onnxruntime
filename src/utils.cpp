#include "utils.h"

std::wstring utils::charToWstring(const char* str)
{
    typedef std::codecvt_utf8<wchar_t> convert_type;
    std::wstring_convert<convert_type, wchar_t> converter;

    return converter.from_bytes(str);
}

std::vector<std::string> utils::loadNames(const std::string& path)
{
    // load class names
    std::vector<std::string> class_names;
    std::ifstream infile(path);
    if (infile.good())
    {
        std::string line;
        while (getline (infile, line))
        {
            class_names.emplace_back(line);
        }
        infile.close();
    }
    else
    {
        std::cerr << ">>> ERROR: Failed to access class name path: " << path
                  << "\n>>>\tDoes the file exist? Permission to read it?\n";
    }

    return class_names;
}

size_t utils::vectorProduct(const std::vector<int64>& vector)
{
    if (vector.empty())
        return -1;

    size_t product = 1;
    for (int64 element : vector)
        product *= element;

    return product;
}

void utils::visualizeDetection(cv::Mat& image, Detection &detection, std::vector<std::string> classNames)
{
    for (int i = 0; i < detection.size(); i++)
    {
        cv::rectangle(image, detection.boxes[i], cv::Scalar(229, 160, 21), 2);

        int x = detection.boxes[i].x;
        int y = detection.boxes[i].y;
        int w = detection.boxes[i].width;
        // int h = boxes[idx].height;

        cv::rectangle(image, cv::Point(x, y - 25), cv::Point(x + w, y), cv::Scalar(229, 160, 21), -1);
        int classId = detection.classIds[i];

        cv::putText(image, classNames[classId], cv::Point(x, y - 5), cv::FONT_ITALIC, 0.8, cv::Scalar(255, 255, 255), 2);
    }
}