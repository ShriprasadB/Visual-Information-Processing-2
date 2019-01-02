#include<iostream>
#include<cmath>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>
#define MASK_SIZE 11
#define PI 3.14

using namespace std;
using namespace cv;

//x gradient
int calculateXGradient(Mat image, int i, int j)
{
	int gx = (image.at<uchar>(i - 1, j - 1) + (2 * image.at<uchar>(i, j - 1)) + image.at<uchar>(i + 1, j - 1)) - (image.at<uchar>(i - 1, j + 1) + (2 * image.at<uchar>(i, j + 1)) + image.at<uchar>(i + 1, j + 1));
	return gx;
}

//y gradient
int calculateYGradient(Mat image, int i, int j)
{
	int gy = (image.at<uchar>(i + 1, j + 1) + (2 * image.at<uchar>(i + 1, j)) + image.at<uchar>(i + 1, j - 1)) - (image.at<uchar>(i - 1, j + 1) + (2 * image.at<uchar>(i - 1, j)) + image.at<uchar>(i - 1, j - 1));
	return gy;
}

//Sobel filter on image
void sobel_operation(Mat img)
{
	Mat final_image;
	int Gx, Gy, G;

	final_image = img.clone();		//replica of original image

	for (int i = 0; i < final_image.rows; i++)	//initialize cloned image
	{
		for (int j = 0; j < final_image.cols; j++)
		{
			final_image.at<uchar>(i, j) = 0.0;
		}
	}

	for (int i = 1; i < final_image.rows - 1; i++)	//We ignore the boundary of the initial image of thickness 1 pixel
	{
		for (int j = 1; j < final_image.cols - 1; j++)
		{
			Gx = calculateXGradient(img, i, j);
			Gy = calculateYGradient(img, i, j);
			G = abs(Gx) + abs(Gy);		//gradient magnitude
			if (G > 255)
			{
				G = 255;
			}
			if (G < 0)
			{
				G = 0;
			}
			final_image.at<uchar>(i, j) = G; //delegate gradient value to each pixel
		}
	}

	namedWindow("Original Image");
	imshow("Original Image", img);	//original image
	namedWindow("resultant Image");
	imshow("resultant Image", final_image);	// final image
	waitKey();
	cvDestroyAllWindows();
}

//Addition of two images
Mat add_images(Mat src, Mat dest)
{
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			dest.at<uchar>(i, j) = src.at<uchar>(i, j) + dest.at<uchar>(i, j);
		}
	}
	return dest;
}

//Subtraction of two images
Mat subtract_images(Mat src, Mat dest)
{
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			dest.at<uchar>(i, j) = src.at<uchar>(i, j) - dest.at<uchar>(i, j);
		}
	}
	return dest;
}

//Unsharp masking
void unsharp_masking(Mat img)
{
	Mat final_image = img.clone();
	int sum;
	for (int i = 0; i < final_image.rows; i++)
	{
		for (int j = 0; j < final_image.cols; j++)
		{ 
			final_image.at<uchar>(i, j) = 0.0;
		}
	}
	//set Gaussian mask
	float GaussianMask[3][3] = 
	{
		{1/16.0, 2/16.0, 1/16.0},
		{2/16.0, 4/16.0, 2/16.0},
		{1/16.0, 2/16.0, 1/16.0}
	};
	//convolution on the image with Gaussian mask
	for (int i = 1; i < final_image.rows-1; i++)
	{
		for (int j = 1; j < final_image.cols-1; j++)
		{
final_image.at<uchar>(i, j) = saturate_cast<uchar>((GaussianMask[2][2] * img.at<uchar>(i - 1, j + 1)) + (GaussianMask[1][2] * img.at<uchar>(i, j + 1)) + (GaussianMask[0][2] * img.at<uchar>(i + 1, j + 1)) + (GaussianMask[2][1] * img.at<uchar>(i - 1, j)) + (GaussianMask[1][1] * img.at<uchar>(i, j)) + (GaussianMask[0][1] * img.at<uchar>(i + 1, j)) + (GaussianMask[2][0] * img.at<uchar>(i - 1, j - 1)) + (GaussianMask[1][0] * img.at<uchar>(i, j - 1)) + (GaussianMask[0][0] * img.at<uchar>(i + 1, j - 1)));
/*
if(final_image.at<uchar>(i, j)>255)
{
final_image.at<uchar>(i, j) = 255;
}
if(final_image.at<uchar>(i, j)<0)
{
final_image.at<uchar>(i, j) = 0;
}*/
		}
	}
	namedWindow("original");
	imshow("original", img);			//Original image
	namedWindow("resultant");
	imshow("resultant", final_image);				//Blurred image using Gaussian mask
	final_image = subtract_images(img, final_image);
	final_image = add_images(img, final_image);
	namedWindow("Final after addition");		//Enhanced image
	imshow("Final after addition", final_image);
	waitKey();
	cvDestroyAllWindows();
}

//Calculates and prints Laplacian of Gaussian mask
void calculate_LoG(Mat img)
{
	int mask[MASK_SIZE][MASK_SIZE];
	float sigma;
	int n;								
	cout << "Enter size of mask: ";
	cin >> n;
	cout << "Enter value of sigma: ";
	cin >> sigma;
	float s2 = sigma * sigma;
	float s4 = sigma * sigma * sigma * sigma;
	int min = int(ceil(-n / 2));
	int max = (ceil(n / 2));
	int l = 0;
	//for sigma equal 5
	if (sigma == 5)
	{
		for (int i = min; i <= max; ++i)
		{
			int m = 0;
			for (int j = min; j <= max; ++j)
			{
float value = (float)(-1.0/(PI * (s4))) * (1.0 - ((i * i + j * j) / (2.0 * (s2)))) * (float)pow(2.7, ((-(i * i + j * j) / (2.0 * (s2)))));
				
				if((l>=2&&l<=7)&&(m>=2&&m<=7))
				{
					mask[l][m] = int(value * 40000);
				}
				else
				{
					mask[l][m] = int(value * 40000*-1);
				}
				++m;
			}
			++l;
		}
	}
	else
	{
		for (int i = min; i <= max; ++i)
		{
			int m = 0;
			for (int j = min; j <= max; ++j)
			{
float value = (float)(-1.0/(PI * (s4))) * (1.0 - ((i * i + j * j) / (2.0 * (s2)))) * (float)pow(2.7, ((-(i * i + j * j) / (2.0 * (s2)))));
				mask[l][m] = int(value * 35000);
				++m;
			}
			++l;
		}
	}
	//Print LoG mask
	cout << endl << "Laplacian of Gaussian:" << endl;
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			cout << mask[i][j] << "\t";
		}
		cout << endl;
	}
	//Initialize final image values
	Mat final_image = img.clone();
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			final_image.at<uchar>(i, j) = 0;
		}
	}
	//Laplacian of Gaussian mask to the image
	for (int x = 0; x < img.rows; x++)
	{
		for (int y = 0; y < img.cols; y++)
		{
			double value = 0;
			for (int s = min; s <= max; s++)
			{
				for (int t = min; t <= max; t++)
				{
					
					value = value + mask[s+max][t+max] * (double)img.at<uchar>(x - s, y - t);
				//	q++;
				}
				//p++;
			}
		//	if (value < 0)
		//	{
		//		value = 0;
		//	}
		//	else
		//	{
		
				final_image.at<uchar>(x,y) = saturate_cast<uchar>(value);
		//	}
		}
	}
/*	Mat src, gray, dst, abs_dst;
	    src = imread( "/home/shri/Downloads/vip2/TestImage/ant_gray.bmp" );	
	GaussianBlur( src, src, Size(3,3), 0, 0, BORDER_DEFAULT );
    cvtColor( src, gray, CV_RGB2GRAY );
 
    /// Apply Laplace function
    Laplacian( gray, dst, CV_16S, 3, 1, 0, BORDER_DEFAULT );
    convertScaleAbs( dst, abs_dst );
    imshow( "result", abs_dst );*/
	namedWindow("Original Image");
	imshow("Original Image", img);
	namedWindow("resultant Image");
	imshow("resultant Image", final_image);
	waitKey();
	cvDestroyAllWindows();

}
int main(int argc, char *argv[])
{
	string filename;
	cout << "Enter image name with full path " << endl;
	cin >> filename;
	Mat img = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);	//Load the image
	namedWindow("Original Image", WINDOW_AUTOSIZE);
	imshow("Original Image", img);
	int ch;
	do
	{
		//menu
cout << endl << "1. Unsharp Masking method" << endl << "2. Sobel operator" << endl << "3. Laplacian of Gaussian" << endl << "4. Exit" << endl << "Enter your choice:";
		cin >> ch;
		switch (ch)
		{
			case 2: sobel_operation(img);
				break;
			case 1: unsharp_masking(img);
				break;
			case 3: calculate_LoG(img);
				break;
			case 4: ch = 4;
				break;
			default: break;
		}
	}
	while (ch != 4);
	waitKey();
	return 0;
}
