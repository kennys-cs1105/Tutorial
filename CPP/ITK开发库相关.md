# ITK开发库算法相关

*Created by KennyS*

---


## 参考博客

[2d-3d CNN](https://blog.csdn.net/weixin_40977054/article/details/112854175)
[ITK读写RAW数据与像素遍历的方法](https://blog.csdn.net/Willfore/article/details/129202425?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522170556680416800226513389%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=170556680416800226513389&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-10-129202425-null-null.142^v99^pc_search_result_base1&utm_term=vs%20drr%20itk&spm=1018.2226.3001.4187)
[win10+visual studio 2022+itk+生成drr](https://blog.csdn.net/sdhdsf132452/article/details/126853325?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522170556680416800226513389%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=170556680416800226513389&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-2-126853325-null-null.142^v99^pc_search_result_base1&utm_term=vs%20drr%20itk&spm=1018.2226.3001.4187)
[liunx环境,python调用ITK（c++版本）批量生成Drr](https://blog.csdn.net/SL1029_/article/details/131598410?ops_request_misc=&request_id=&biz_id=102&utm_term=vs%20drr%20itk&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-1-131598410.142^v99^pc_search_result_base1&spm=1018.2226.3001.4187)


## Linux环境下ITK安装

- ITK5.3.tar.gz, 官网download
- 编译
    ```bash
    cd /usr/local/itk
    sudo tar xvzf InsightToolkit-5.2.1.tar.gz

    cd /usr/local/itk/InsightToolkit-5.2.1
    sudo mkdir build
    cd build
    sudo cmake -DITK_USE_REVIEW=ON ..

    sudo make -j4
    sudo make install
    ```

- 测试编译
    ```bash
    cd /home
    mkdir hello

    cp /home/ITK/InsightToolKit-4.10.0/Examples/Installation/CMakeLists.txt /home/hello
    cp /home/ITK/InsightToolKit-4.10.0/Examples/Installation/HelloWorld.cxx /home/hello
    cd /home/hello
    mkdir build

    cd /home/hello/build
    cmake ../../hello

    cd /home/hello/build
    make

    ./HelloWorld
    ```


## DRR算法

1. CMakeList.txt

```CMake
# This is the root ITK CMakeLists file.
cmake_minimum_required(VERSION 3.16.3 FATAL_ERROR)
foreach(p
    ## Only policies introduced after the cmake_minimum_required
    ## version need to explicitly be set to NEW.
    CMP0070 #3.10.0 Define ``file(GENERATE)`` behavior for relative paths.
    CMP0071 #3.10.0 Let ``AUTOMOC`` and ``AUTOUIC`` process ``GENERATED`` files.
    )
  if(POLICY ${p})
    cmake_policy(SET ${p} NEW)
  endif()
endforeach()


# This project is designed to be built outside the Insight source tree.
project(drr)

#set(ITK_DIR "/usr/local/itk/InsightToolkit-5.3.0/build/lib/cmake/ITK-5.3")
# Find ITK.
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

add_executable(drr main.cpp)

target_link_libraries(drr ${ITK_LIBRARIES})
```

2. main.cpp

```cpp
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkResampleImageFilter.h"
#include "itkCenteredEuler3DTransform.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkRescaleIntensityImageFilter.h"
#include <itkGDCMImageIO.h>
#include <itkPNGImageIO.h>
#include "itkPNGImageIOFactory.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkImageSeriesReader.h"
#include "itkImageSeriesWriter.h"
#include "itkNumericSeriesFileNames.h"
#include "itkGiplImageIOFactory.h"
#include "itkCastImageFilter.h"
#include <itkNiftiImageIO.h>
#include <itkNiftiImageIOFactory.h>
#include "itkMetaImageIOFactory.h"
#pragma comment(lib,"ws2_32.lib")
#pragma comment(lib,"Rpcrt4.lib")
#pragma  once
#pragma  comment(lib,"Psapi.lib")
// Software Guide : BeginCodeSnippet
#include "itkRayCastInterpolateImageFunction.h"
#include <iostream>
// Software Guide : EndCodeSnippet
#include <sys/io.h>
using namespace std;

int main(int argc, char* argv[])
{
    int file_list[1018];
    for (int i = 1; i < 1018; i++) {
        file_list[i] = i;
        //cout << i << endl;
    }
    // ×Ü¹²1013žöÎÄŒþ£¬È»ºóÀÁµÃŽóžÄŽúÂëÁË¡£Èç¹ûÎÄŒþÊýÓÐ²»Í¬ŒÇµÃžùŸÝ×ÔŒºµÄÎÄŒþœøÐÐÐÞžÄ
    // Êä³öµÄÎÄŒþ»á±£ŽæÔÚÎÄŒþÏÂÎªdrrÎÄŒþŒÐÏÂ£¬Îª.pngÊýŸÝÀàÐÍ
    for (int i = 1; i < 1018; i++) {
        int number = file_list[i];
        string num = to_string(number);

        string in_drr = "/home/kennys/drr/data/ct/";
        string out_drr = "/home/kennys/drr/data/drr/";

        string input = in_drr + num;
        string output = out_drr + num + ".png";

        const char* input_name = input.c_str();
        const char* output_name = output.c_str();
        cout << input_name << endl;
        //cout << output_name << endl;

        //const char* input_name = "D:/7_15data/ct/1 ";
        //const char* output_name = "D:/7_15data/drr/5.png";

        bool ok;  //true
        bool verbose = false; //true

        // CT volume rotation around isocenter along x,y,z axis in degrees
        float rx = 90;
        float ry = 0;
        float rz = 90.; //œÇ¶Èµ÷œÚ
        //float rz = -90.;//²à×ÅµÄ
        //float rz = 0.; //Õý×ÅµÄ
        //float rz = 180.; //±³Ãæ

        // Translation parameter of the isocenter in mm
        float tx = 0.;
        float ty = 0.;
        float tz = 0.;

        // The pixel indices of the isocenter
        // The pixel indices of the isocenter
        float cx = 0.;
        float cy = 0.;
        float cz = 0.;
        //1000.
        //float sid = 3000; //400. Source to isocenter distance in mm ÔŽµœµÈÖÐÐÄŸà 1000
        float sid = 500; //400. Source to isocenter distance in mm ÔŽµœµÈÖÐÐÄŸà 1000

        //Default pixel spacing in the iso-center plane in mm
        //µÈœÇµãÆœÃæÖÐµÄÄ¬ÈÏÏñËØŒäŸà£¬ÒÔºÁÃ×Îªµ¥Î»
        //float sx = 2.5; //1.  2,5
        //float sy = 0.5; //1.
        float sx = 2.5; //1.
        float sy = 2.5; //1.

        // Size of the output image in number of pixels
        int dx = 256;
        int dy = 256;

        // The central axis positions of the 2D images in continuous indices
        //Á¬ÐøË÷ÒýÖÐ2DÍŒÏñµÄÖÐÐÄÖáÎ»ÖÃ
        float o2Dx = 0;
        float o2Dy = 0;

        double threshold = -600; //1


        const     unsigned int   Dimension = 3;

        typedef  float    InputPixelType;
        typedef  unsigned char  OutputPixelType;


        typedef itk::Image< InputPixelType, Dimension >   InputImageType;
        typedef itk::Image< OutputPixelType, Dimension >   OutputImageType;
        //ÉèÖÃÊä³öÀàÐÍ

        InputImageType::Pointer image;


        if (input_name)
        {
            


            typedef itk::ImageFileReader< InputImageType >  ReaderType;
            ReaderType::Pointer reader = ReaderType::New();
            itk::NiftiImageIOFactory::RegisterOneFactory();
            reader->SetFileName(input_name);



            try
            {
                reader->Update();
            }
            catch (itk::ExceptionObject& err)
            {
                std::cerr << "ERROR: ExceptionObject caught !" << std::endl;
                std::cerr << err << std::endl;
                return EXIT_FAILURE;
            }

            image = reader->GetOutput();

        }
        else
        {   // No input image specified so create a cube

            image = InputImageType::New();

            InputImageType::SpacingType spacing;
            spacing[0] = 3.;
            spacing[1] = 3.;
            spacing[2] = 3.;
            image->SetSpacing(spacing);

            InputImageType::PointType origin;
            origin[0] = 0.;
            origin[1] = 0.;
            origin[2] = 0.;
            image->SetOrigin(origin);

            InputImageType::IndexType start;

            start[0] = 0;  // first index on X
            start[1] = 0;  // first index on Y
            start[2] = 0;  // first index on Z

            InputImageType::SizeType  size;

            size[0] = 61;  // size along X
            size[1] = 61;  // size along Y
            size[2] = 61;  // size along Z

            InputImageType::RegionType region;

            region.SetSize(size);
            region.SetIndex(start);

            image->SetRegions(region);
            image->Allocate(true); // initialize to zero.

            image->Update();

            typedef itk::ImageRegionIteratorWithIndex< InputImageType > IteratorType;

            IteratorType iterate(image, image->GetLargestPossibleRegion());

            while (!iterate.IsAtEnd())
            {

                InputImageType::IndexType idx = iterate.GetIndex();

                if ((idx[0] >= 6) && (idx[0] <= 54)
                    && (idx[1] >= 6) && (idx[1] <= 54)
                    && (idx[2] >= 6) && (idx[2] <= 54)

                    && ((((idx[0] <= 11) || (idx[0] >= 49))
                         && ((idx[1] <= 11) || (idx[1] >= 49)))

                        || (((idx[0] <= 11) || (idx[0] >= 49))
                            && ((idx[2] <= 11) || (idx[2] >= 49)))

                        || (((idx[1] <= 11) || (idx[1] >= 49))
                            && ((idx[2] <= 11) || (idx[2] >= 49)))))
                {
                    iterate.Set(10);
                }

                else if ((idx[0] >= 18) && (idx[0] <= 42)
                         && (idx[1] >= 18) && (idx[1] <= 42)
                         && (idx[2] >= 18) && (idx[2] <= 42)

                         && ((((idx[0] <= 23) || (idx[0] >= 37))
                              && ((idx[1] <= 23) || (idx[1] >= 37)))

                             || (((idx[0] <= 23) || (idx[0] >= 37))
                                 && ((idx[2] <= 23) || (idx[2] >= 37)))

                             || (((idx[1] <= 23) || (idx[1] >= 37))
                                 && ((idx[2] <= 23) || (idx[2] >= 37)))))
                {
                    iterate.Set(60);
                }

                else if ((idx[0] == 30) && (idx[1] == 30) && (idx[2] == 30))
                {
                    iterate.Set(100);
                }

                ++iterate;
            }



#ifdef WRITE_CUBE_IMAGE_TO_FILE
            const char* filename = "cube.gipl";
                        typedef itk::ImageFileWriter< InputImageType >  WriterType;
                        WriterType::Pointer writer = WriterType::New();
                        itk::GiplImageIOFactory::RegisterOneFactory();
                        writer->SetFileName(filename);
                        writer->SetInput(image);

                        try
                        {
                                std::cout << "Writing image: " << filename << std::endl;
                                writer->Update();
                        }
                        catch (itk::ExceptionObject& err)
                        {

                                std::cerr << "ERROR: ExceptionObject caught !" << std::endl;
                                std::cerr << err << std::endl;
                                return EXIT_FAILURE;
                        }
#endif
        }


        // Print out the details of the input volume

        if (verbose)
        {
            unsigned int i;
            const InputImageType::SpacingType spacing = image->GetSpacing();
            std::cout << std::endl << "Input ";

            InputImageType::RegionType region = image->GetBufferedRegion();
            region.Print(std::cout);

            std::cout << "  Resolution: [";
            for (i = 0; i < Dimension; i++)
            {
                std::cout << spacing[i];
                if (i < Dimension - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;

            const InputImageType::PointType origin = image->GetOrigin();
            std::cout << "  Origin: [";
            for (i = 0; i < Dimension; i++)
            {
                std::cout << origin[i];
                if (i < Dimension - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl << std::endl;
        }

        typedef itk::ResampleImageFilter<InputImageType, InputImageType > FilterType;

        FilterType::Pointer filter = FilterType::New();

        filter->SetInput(image);
        filter->SetDefaultPixelValue(0);

        typedef itk::CenteredEuler3DTransform< double >  TransformType;

        TransformType::Pointer transform = TransformType::New();

        transform->SetComputeZYX(true);

        TransformType::OutputVectorType translation;

        translation[0] = tx;
        translation[1] = ty;
        translation[2] = tz;

        // constant for converting degrees into radians
        const double dtr = (std::atan(1.0) * 4.0) / 180.0;

        transform->SetTranslation(translation);
        transform->SetRotation(dtr * rx, dtr * ry, dtr * rz);

        InputImageType::PointType   imOrigin = image->GetOrigin();
        InputImageType::SpacingType imRes = image->GetSpacing();

        typedef InputImageType::RegionType     InputImageRegionType;
        typedef InputImageRegionType::SizeType InputImageSizeType;

        InputImageRegionType imRegion = image->GetBufferedRegion();
        InputImageSizeType   imSize = imRegion.GetSize();

        imOrigin[0] += imRes[0] * static_cast<double>(imSize[0]) / 2.0;
        imOrigin[1] += imRes[1] * static_cast<double>(imSize[1]) / 2.0;
        imOrigin[2] += imRes[2] * static_cast<double>(imSize[2]) / 2.0;

        TransformType::InputPointType center;
        center[0] = cx + imOrigin[0];
        center[1] = cy + imOrigin[1];
        center[2] = cz + imOrigin[2];

        transform->SetCenter(center);

        if (verbose)
        {
            std::cout << "Image size: "
                      << imSize[0] << ", " << imSize[1] << ", " << imSize[2]
                      << std::endl << "   resolution: "
                      << imRes[0] << ", " << imRes[1] << ", " << imRes[2]
                      << std::endl << "   origin: "
                      << imOrigin[0] << ", " << imOrigin[1] << ", " << imOrigin[2]
                      << std::endl << "   center: "
                      << center[0] << ", " << center[1] << ", " << center[2]
                      << std::endl << "Transform: " << transform << std::endl;
        }

        typedef itk::RayCastInterpolateImageFunction<InputImageType, double>
                InterpolatorType;
        InterpolatorType::Pointer interpolator = InterpolatorType::New();
        interpolator->SetTransform(transform);

        //
        // We can then specify a threshold above which the volume's
        // intensities will be integrated.

        interpolator->SetThreshold(threshold);

        InterpolatorType::InputPointType focalpoint;

        focalpoint[0] = imOrigin[0];
        focalpoint[1] = imOrigin[1];
        focalpoint[2] = imOrigin[2] - sid / 2.;

        interpolator->SetFocalPoint(focalpoint);
        // Software Guide : EndCodeSnippet

        if (verbose)
        {
            std::cout << "Focal Point: "
                      << focalpoint[0] << ", "
                      << focalpoint[1] << ", "
                      << focalpoint[2] << std::endl;
        }

        // Software Guide : BeginLatex
        //
        // Having initialised the interpolator we pass the object to the
        // resample filter.

        interpolator->Print(std::cout);

        filter->SetInterpolator(interpolator);
        filter->SetTransform(transform);


        // setup the scene
        InputImageType::SizeType   size;

        size[0] = dx;  // number of pixels along X of the 2D DRR image
        size[1] = dy;  // number of pixels along Y of the 2D DRR image
        size[2] = 1;   // only one slice

        filter->SetSize(size);

        InputImageType::SpacingType spacing;

        spacing[0] = sx;  // pixel spacing along X of the 2D DRR image [mm]
        spacing[1] = sy;  // pixel spacing along Y of the 2D DRR image [mm]
        spacing[2] = 1.0; // slice thickness of the 2D DRR image [mm]

        filter->SetOutputSpacing(spacing);

        // Software Guide : EndCodeSnippet

        if (verbose)
        {
            std::cout << "Output image size: "
                      << size[0] << ", "
                      << size[1] << ", "
                      << size[2] << std::endl;

            std::cout << "Output image spacing: "
                      << spacing[0] << ", "
                      << spacing[1] << ", "
                      << spacing[2] << std::endl;
        }



        double origin[Dimension];

        origin[0] = imOrigin[0] + o2Dx - sx * ((double)dx - 1.) / 2.;
        origin[1] = imOrigin[1] + o2Dy - sy * ((double)dy - 1.) / 2.;
        origin[2] = imOrigin[2] + sid / 2.;

        filter->SetOutputOrigin(origin);
        // Software Guide : EndCodeSnippet

        if (verbose)
        {
            std::cout << "Output image origin: "
                      << origin[0] << ", "
                      << origin[1] << ", "
                      << origin[2] << std::endl;
        }

        // create writer

        if (output_name)
        {

            typedef itk::RescaleIntensityImageFilter<
                    InputImageType, OutputImageType > RescaleFilterType;
            RescaleFilterType::Pointer rescaler = RescaleFilterType::New();
            rescaler->SetOutputMinimum(0);
            rescaler->SetOutputMaximum(255);
            rescaler->SetInput(filter->GetOutput());
            //
            typedef itk::ImageFileWriter< OutputImageType >  WriterType;
            WriterType::Pointer writer = WriterType::New();

            typedef itk::PNGImageIO pngType;
            pngType::Pointer pngIO1 = pngType::New();
            itk::PNGImageIOFactory::RegisterOneFactory();
            writer->SetFileName(output_name);
            writer->SetImageIO(pngIO1);
            writer->SetImageIO(itk::PNGImageIO::New());
            writer->SetInput(rescaler->GetOutput());

            try
            {
                std::cout << "Writing image: " << output_name << std::endl;
                writer->Update();
            }
            catch (itk::ExceptionObject& err)
            {
                std::cerr << "ERROR: ExceptionObject caught !" << std::endl;
                std::cerr << err << std::endl;
            }

        }
        else
        {
            filter->Update();
        }
        /*        system("pause");
            return EXIT_SUCCESS;*/
    }
}
```

3. 设置data

4. 编译运行

```bash
cmake .
make
./drr
```


## DRR算法生成动态库静态库可执行文件

1. DRR_example

```cpp
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkResampleImageFilter.h"
#include "itkCenteredEuler3DTransform.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkImageRegionIteratorWithIndex.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkRayCastInterpolateImageFunction.h"


void
usage()
{
  std::cerr << "\n";
  std::cerr << "Usage: DRR <options> [input]\n";
  std::cerr << "  calculates the Digitally Reconstructed Radiograph from a "
               "volume. \n\n";
  std::cerr << " where <options> is one or more of the following:\n\n";
  std::cerr << "  <-h>                    Display (this) usage information\n";
  std::cerr << "  <-v>                    Verbose output [default: no]\n";
  std::cerr << "  <-res float float>      Pixel spacing of the output image "
               "[default: "
               "1x1mm]  \n";
  std::cerr << "  <-size int int>         Dimension of the output image "
               "[default: 501x501]  \n";
  std::cerr
    << "  <-sid float>            Distance of ray source (focal point) "
       "[default: 400mm]\n";
  std::cerr
    << "  <-t float float float>  Translation parameter of the camera \n";
  std::cerr
    << "  <-rx float>             Rotation around x,y,z axis in degrees \n";
  std::cerr << "  <-ry float>\n";
  std::cerr << "  <-rz float>\n";
  std::cerr << "  <-normal float float>   The 2D projection normal position "
               "[default: 0x0mm]\n";
  std::cerr
    << "  <-cor float float float> The centre of rotation relative to centre "
       "of volume\n";
  std::cerr << "  <-threshold float>      Threshold [default: 0]\n";
  std::cerr << "  <-o file>               Output image filename\n\n";
  std::cerr << "                          by  thomas@hartkens.de\n";
  std::cerr << "                          and john.hipwell@kcl.ac.uk (CISG "
               "London)\n\n";
  exit(1);
}

int
main(int argc, char * argv[])
{
  char * input_name = nullptr;
  char * output_name = nullptr;

  bool ok;
  bool verbose = false;

  float rx = 0.;
  float ry = 0.;
  float rz = 0.;

  float tx = 0.;
  float ty = 0.;
  float tz = 0.;

  float cx = 0.;
  float cy = 0.;
  float cz = 0.;

  float sid = 400.;

  float sx = 1.;
  float sy = 1.;

  int dx = 501;
  int dy = 501;

  float o2Dx = 0;
  float o2Dy = 0;

  double threshold = 0;


  // Parse command line parameters

  while (argc > 1)
  {
    ok = false;

    if ((ok == false) && (strcmp(argv[1], "-h") == 0))
    {
      argc--;
      argv++;
      ok = true;
      usage();
    }

    if ((ok == false) && (strcmp(argv[1], "-v") == 0))
    {
      argc--;
      argv++;
      ok = true;
      verbose = true;
    }

    if ((ok == false) && (strcmp(argv[1], "-rx") == 0))
    {
      argc--;
      argv++;
      ok = true;
      rx = std::stod(argv[1]);
      argc--;
      argv++;
    }

    if ((ok == false) && (strcmp(argv[1], "-ry") == 0))
    {
      argc--;
      argv++;
      ok = true;
      ry = std::stod(argv[1]);
      argc--;
      argv++;
    }

    if ((ok == false) && (strcmp(argv[1], "-rz") == 0))
    {
      argc--;
      argv++;
      ok = true;
      rz = std::stod(argv[1]);
      argc--;
      argv++;
    }

    if ((ok == false) && (strcmp(argv[1], "-threshold") == 0))
    {
      argc--;
      argv++;
      ok = true;
      threshold = std::stod(argv[1]);
      argc--;
      argv++;
    }

    if ((ok == false) && (strcmp(argv[1], "-t") == 0))
    {
      argc--;
      argv++;
      ok = true;
      tx = std::stod(argv[1]);
      argc--;
      argv++;
      ty = std::stod(argv[1]);
      argc--;
      argv++;
      tz = std::stod(argv[1]);
      argc--;
      argv++;
    }

    if ((ok == false) && (strcmp(argv[1], "-cor") == 0))
    {
      argc--;
      argv++;
      ok = true;
      cx = std::stod(argv[1]);
      argc--;
      argv++;
      cy = std::stod(argv[1]);
      argc--;
      argv++;
      cz = std::stod(argv[1]);
      argc--;
      argv++;
    }

    if ((ok == false) && (strcmp(argv[1], "-res") == 0))
    {
      argc--;
      argv++;
      ok = true;
      sx = std::stod(argv[1]);
      argc--;
      argv++;
      sy = std::stod(argv[1]);
      argc--;
      argv++;
    }

    if ((ok == false) && (strcmp(argv[1], "-size") == 0))
    {
      argc--;
      argv++;
      ok = true;
      dx = std::stoi(argv[1]);
      argc--;
      argv++;
      dy = std::stoi(argv[1]);
      argc--;
      argv++;
    }

    if ((ok == false) && (strcmp(argv[1], "-sid") == 0))
    {
      argc--;
      argv++;
      ok = true;
      sid = std::stod(argv[1]);
      argc--;
      argv++;
    }

    if ((ok == false) && (strcmp(argv[1], "-normal") == 0))
    {
      argc--;
      argv++;
      ok = true;
      o2Dx = std::stod(argv[1]);
      argc--;
      argv++;
      o2Dy = std::stod(argv[1]);
      argc--;
      argv++;
    }

    if ((ok == false) && (strcmp(argv[1], "-o") == 0))
    {
      argc--;
      argv++;
      ok = true;
      output_name = argv[1];
      argc--;
      argv++;
    }

    if (ok == false)
    {

      if (input_name == nullptr)
      {
        input_name = argv[1];
        argc--;
        argv++;
      }

      else
      {
        std::cerr << "ERROR: Can not parse argument " << argv[1] << std::endl;
        usage();
      }
    }
  }

  if (verbose)
  {
    if (input_name)
      std::cout << "Input image: " << input_name << std::endl;
    if (output_name)
      std::cout << "Output image: " << output_name << std::endl;
  }

  constexpr unsigned int Dimension = 3;
  using InputPixelType = short;
  using OutputPixelType = unsigned char;

  using InputImageType = itk::Image<InputPixelType, Dimension>;
  using OutputImageType = itk::Image<OutputPixelType, Dimension>;

  InputImageType::Pointer image;

  if (input_name)
  {

    using ReaderType = itk::ImageFileReader<InputImageType>;
    ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName(input_name);


    try
    {
      reader->Update();
    }
    catch (const itk::ExceptionObject & err)
    {
      std::cerr << "ERROR: ExceptionObject caught !" << std::endl;
      std::cerr << err << std::endl;
      return EXIT_FAILURE;
    }

    image = reader->GetOutput();
  }
  else
  { // No input image specified so create a cube

    image = InputImageType::New();

    InputImageType::SpacingType spacing;
    spacing[0] = 3.;
    spacing[1] = 3.;
    spacing[2] = 3.;
    image->SetSpacing(spacing);

    InputImageType::PointType origin;
    origin[0] = 0.;
    origin[1] = 0.;
    origin[2] = 0.;
    image->SetOrigin(origin);

    InputImageType::IndexType start;

    start[0] = 0; // first index on X
    start[1] = 0; // first index on Y
    start[2] = 0; // first index on Z

    InputImageType::SizeType size;

    size[0] = 61; // size along X
    size[1] = 61; // size along Y
    size[2] = 61; // size along Z

    InputImageType::RegionType region;

    region.SetSize(size);
    region.SetIndex(start);

    image->SetRegions(region);
    image->Allocate(true); // initialize to zero.

    image->Update();

    using IteratorType = itk::ImageRegionIteratorWithIndex<InputImageType>;

    IteratorType iterate(image, image->GetLargestPossibleRegion());

    while (!iterate.IsAtEnd())
    {

      InputImageType::IndexType idx = iterate.GetIndex();

      if ((idx[0] >= 6) && (idx[0] <= 54) && (idx[1] >= 6) &&
          (idx[1] <= 54) && (idx[2] >= 6) && (idx[2] <= 54)

          && ((((idx[0] <= 11) || (idx[0] >= 49)) &&
               ((idx[1] <= 11) || (idx[1] >= 49)))

              || (((idx[0] <= 11) || (idx[0] >= 49)) &&
                  ((idx[2] <= 11) || (idx[2] >= 49)))

              || (((idx[1] <= 11) || (idx[1] >= 49)) &&
                  ((idx[2] <= 11) || (idx[2] >= 49)))))
      {
        iterate.Set(10);
      }

      else if ((idx[0] >= 18) && (idx[0] <= 42) && (idx[1] >= 18) &&
               (idx[1] <= 42) && (idx[2] >= 18) && (idx[2] <= 42)

               && ((((idx[0] <= 23) || (idx[0] >= 37)) &&
                    ((idx[1] <= 23) || (idx[1] >= 37)))

                   || (((idx[0] <= 23) || (idx[0] >= 37)) &&
                       ((idx[2] <= 23) || (idx[2] >= 37)))

                   || (((idx[1] <= 23) || (idx[1] >= 37)) &&
                       ((idx[2] <= 23) || (idx[2] >= 37)))))
      {
        iterate.Set(60);
      }

      else if ((idx[0] == 30) && (idx[1] == 30) && (idx[2] == 30))
      {
        iterate.Set(100);
      }

      ++iterate;
    }


#ifdef WRITE_CUBE_IMAGE_TO_FILE
    const char * filename = "cube.gipl";
    using WriterType = itk::ImageFileWriter<InputImageType>;
    WriterType::Pointer writer = WriterType::New();

    writer->SetFileName(filename);
    writer->SetInput(image);

    try
    {
      std::cout << "Writing image: " << filename << std::endl;
      writer->Update();
    }
    catch (const itk::ExceptionObject & err)
    {

      std::cerr << "ERROR: ExceptionObject caught !" << std::endl;
      std::cerr << err << std::endl;
      return EXIT_FAILURE;
    }
#endif
  }


  // Print out the details of the input volume

  if (verbose)
  {
    unsigned int                      i;
    const InputImageType::SpacingType spacing = image->GetSpacing();
    std::cout << std::endl << "Input ";

    InputImageType::RegionType region = image->GetBufferedRegion();
    region.Print(std::cout);

    std::cout << "  Resolution: [";
    for (i = 0; i < Dimension; i++)
    {
      std::cout << spacing[i];
      if (i < Dimension - 1)
        std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    const InputImageType::PointType origin = image->GetOrigin();
    std::cout << "  Origin: [";
    for (i = 0; i < Dimension; i++)
    {
      std::cout << origin[i];
      if (i < Dimension - 1)
        std::cout << ", ";
    }
    std::cout << "]" << std::endl << std::endl;
  }

  using FilterType = itk::ResampleImageFilter<InputImageType, InputImageType>;

  FilterType::Pointer filter = FilterType::New();

  filter->SetInput(image);
  filter->SetDefaultPixelValue(0);

  using TransformType = itk::CenteredEuler3DTransform<double>;

  TransformType::Pointer transform = TransformType::New();

  transform->SetComputeZYX(true);

  TransformType::OutputVectorType translation;

  translation[0] = tx;
  translation[1] = ty;
  translation[2] = tz;

  // constant for converting degrees into radians
  const double dtr = (std::atan(1.0) * 4.0) / 180.0;

  transform->SetTranslation(translation);
  transform->SetRotation(dtr * rx, dtr * ry, dtr * rz);

  InputImageType::PointType   imOrigin = image->GetOrigin();
  InputImageType::SpacingType imRes = image->GetSpacing();

  using InputImageRegionType = InputImageType::RegionType;
  using InputImageSizeType = InputImageRegionType::SizeType;

  InputImageRegionType imRegion = image->GetBufferedRegion();
  InputImageSizeType   imSize = imRegion.GetSize();

  imOrigin[0] += imRes[0] * static_cast<double>(imSize[0]) / 2.0;
  imOrigin[1] += imRes[1] * static_cast<double>(imSize[1]) / 2.0;
  imOrigin[2] += imRes[2] * static_cast<double>(imSize[2]) / 2.0;

  TransformType::InputPointType center;
  center[0] = cx + imOrigin[0];
  center[1] = cy + imOrigin[1];
  center[2] = cz + imOrigin[2];

  transform->SetCenter(center);

  if (verbose)
  {
    std::cout << "Image size: " << imSize[0] << ", " << imSize[1] << ", "
              << imSize[2] << std::endl
              << "   resolution: " << imRes[0] << ", " << imRes[1] << ", "
              << imRes[2] << std::endl
              << "   origin: " << imOrigin[0] << ", " << imOrigin[1] << ", "
              << imOrigin[2] << std::endl
              << "   center: " << center[0] << ", " << center[1] << ", "
              << center[2] << std::endl
              << "Transform: " << transform << std::endl;
  }

  using InterpolatorType =
    itk::RayCastInterpolateImageFunction<InputImageType, double>;
  InterpolatorType::Pointer interpolator = InterpolatorType::New();
  interpolator->SetTransform(transform);

  interpolator->SetThreshold(threshold);

  InterpolatorType::InputPointType focalpoint;

  focalpoint[0] = imOrigin[0];
  focalpoint[1] = imOrigin[1];
  focalpoint[2] = imOrigin[2] - sid / 2.;

  interpolator->SetFocalPoint(focalpoint);
  // Software Guide : EndCodeSnippet

  if (verbose)
  {
    std::cout << "Focal Point: " << focalpoint[0] << ", " << focalpoint[1]
              << ", " << focalpoint[2] << std::endl;
  }

  interpolator->Print(std::cout);

  filter->SetInterpolator(interpolator);
  filter->SetTransform(transform);

  InputImageType::SizeType size;

  size[0] = dx; // number of pixels along X of the 2D DRR image
  size[1] = dy; // number of pixels along Y of the 2D DRR image
  size[2] = 1;  // only one slice

  filter->SetSize(size);

  InputImageType::SpacingType spacing;

  spacing[0] = sx;  // pixel spacing along X of the 2D DRR image [mm]
  spacing[1] = sy;  // pixel spacing along Y of the 2D DRR image [mm]
  spacing[2] = 1.0; // slice thickness of the 2D DRR image [mm]

  filter->SetOutputSpacing(spacing);

  // Software Guide : EndCodeSnippet

  if (verbose)
  {
    std::cout << "Output image size: " << size[0] << ", " << size[1] << ", "
              << size[2] << std::endl;

    std::cout << "Output image spacing: " << spacing[0] << ", " << spacing[1]
              << ", " << spacing[2] << std::endl;
  }


  double origin[Dimension];

  origin[0] = imOrigin[0] + o2Dx - sx * ((double)dx - 1.) / 2.;
  origin[1] = imOrigin[1] + o2Dy - sy * ((double)dy - 1.) / 2.;
  origin[2] = imOrigin[2] + sid / 2.;

  filter->SetOutputOrigin(origin);
  // Software Guide : EndCodeSnippet

  if (verbose)
  {
    std::cout << "Output image origin: " << origin[0] << ", " << origin[1]
              << ", " << origin[2] << std::endl;
  }

  // create writer

  if (output_name)
  {


    using RescaleFilterType =
      itk::RescaleIntensityImageFilter<InputImageType, OutputImageType>;
    RescaleFilterType::Pointer rescaler = RescaleFilterType::New();
    rescaler->SetOutputMinimum(0);
    rescaler->SetOutputMaximum(255);
    rescaler->SetInput(filter->GetOutput());

    using WriterType = itk::ImageFileWriter<OutputImageType>;
    WriterType::Pointer writer = WriterType::New();

    writer->SetFileName(output_name);
    writer->SetInput(rescaler->GetOutput());

    try
    {
      std::cout << "Writing image: " << output_name << std::endl;
      writer->Update();
    }
    catch (const itk::ExceptionObject & err)
    {
      std::cerr << "ERROR: ExceptionObject caught !" << std::endl;
      std::cerr << err << std::endl;
    }
    // Software Guide : EndCodeSnippet
  }
  else
  {
    filter->Update();
  }

  return EXIT_SUCCESS;
}
```

2. CMakeList.txt

```CMake
cmake_minimum_required(VERSION 3.12)
project(drr_lib)

# 查找ITK
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

# 添加源文件
set(SOURCE_FILES main.cpp)

# 生成动态库
add_library(drr_lib SHARED ${SOURCE_FILES})

# 生成静态库
add_library(drr_lib_static STATIC ${SOURCE_FILES})

# 设置库的输出名称
set_target_properties(drr_lib PROPERTIES OUTPUT_NAME "drr_lib")
set_target_properties(drr_lib_static PROPERTIES OUTPUT_NAME "drr_lib_static")

# 设置库的版本号
set_target_properties(drr_lib PROPERTIES VERSION 1.0)
set_target_properties(drr_lib PROPERTIES SOVERSION 1)

# 指定头文件目录
target_include_directories(drr_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${ITK_INCLUDE_DIRS})

# 链接ITK库
target_link_libraries(drr_lib ${ITK_LIBRARIES})

# 指定静态库输出目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)

# 指定动态库输出目录
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)

# 生成可执行文件并链接库
add_executable(drr main.cpp)
target_link_libraries(drr drr_lib)

# 设置可执行文件输出目录
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)

# 安装目标（可选）
install(TARGETS drr_lib drr_lib_static drr
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)
```

3. 导入环境变量

方式1: `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/kangshuai/drr_lib/build`

方式2: 

```
vim ~/.bashrc
export LD_LIBRARY_PATH=/data/kangshuai/drr_lib/build:$LD_LIBRARY_PATH
source ~/.bashrc
```


## DCM文件转化为NII数据

1. 步骤

- 设置图像数据类型；
- 设置读取模块并指定为dicom格式；
- 根据迭代器将单张dicom传入给读取模块；
- 把数据写到文件夹

```cpp
#include "itkGDCMSeriesFileNames.h"
#include "itkGDCMImageIO.h"
#include <itkImage.h>
#include <itkImageFileReader.h>
#include "itkImageFileWriter.h"
#include "itkImageSeriesReader.h"

// 设置图像数据类型和维度
using PixelType = signed short;
const unsigned int      Dimension = 3;
using ImageType = itk::Image< PixelType, Dimension >;

// 设置读取模块，并指定为读取 dicom
using ReaderType = itk::ImageSeriesReader< ImageType >; //读序列图片
using ImageIOType = itk::GDCMImageIO; //读DICOM图片
ReaderType::Pointer itkReader = ReaderType::New();
ImageIOType::Pointer dicomIO = ImageIOType::New();
itkReader->SetImageIO( dicomIO );//数据读入内存

// 创建 dicom 路径加载迭代器，并设置文件夹路径
using NamesGeneratorType = itk::GDCMSeriesFileNames;
NamesGeneratorType::Pointer nameGenerator = NamesGeneratorType::New();
nameGenerator->SetDirectory(input_path);//设置文件目录

using SeriesIdContainer = std::vector< std::string >;
const SeriesIdContainer & seriesUID = nameGenerator->GetSeriesUIDs();
std::string seriesIdentifier;
seriesIdentifier = seriesUID.begin()->c_str();//通过迭代器读取所有单张切片
typedef std::vector< std::string >   FileNamesContainer;
FileNamesContainer fileNames;
fileNames = nameGenerator->GetFileNames( seriesIdentifier );
itkReader->SetFileNames( fileNames );
itkReader->Update();
dicomIO->GetMetaDataDictionary();//获取DIOCM头文件中信息

// 写出图像
using writeType = itk::ImageFileWriter<ImageType>;
auto writer = writeType::New();
writer->SetFileName(save_path);
writer->SetInput(itkReader->GetOutput());
writer->Update();
```