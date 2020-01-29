#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <linux/videodev2.h>
#include <linux/ioctl.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>



using namespace cv;
using namespace std;

/*
 *Available resolutions are
 *     1920 x 1080
 */

int width = 1920;
int height = 1080;

double K[9] = { 278.8518, 0, 323.7384172, 0 ,
				273.17860196, 225.88590556,
				0, 0, 1};
double D[4] = {-0.32653103,  0.08570291, -0.00523793, -0.00647632};

Mat cameraMatrix = Mat(3, 3, CV_64FC1, &K);
Mat distCoeffs = Mat(4, 1, CV_64FC1, &D);
Mat frame_bayer_16bit(height, width, CV_16UC1);
cuda::GpuMat frame_bayer_16bit_gpu(height, width, CV_16UC1);
cuda::GpuMat frame_bayer_8bit_gpu(height, width, CV_8UC1);
cuda::GpuMat frame_bgr_8bit_gpu(height, width, CV_8UC3);

bool debug_frame = 1;

int main(){
	cout<<"Initialization of Camera"<<endl;
	int fd;
	fd = open("/dev/video1", O_RDWR);
	if (fd < 0){
		fd = open("/dev/video4", O_RDWR);
		//perror("Failed to open device, OPEN");
	}

	// Retrieve the device's capabilities
	v4l2_capability cap;
	if (ioctl(fd, VIDIOC_QUERYCAP, &cap) < 0){
		perror("Failed to get device capabilities, VIDIOC_QUERYCAP");
		return 1;
	}

	// Setup video format
	v4l2_format format;
	format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	format.fmt.pix.pixelformat = V4L2_PIX_FMT_SBGGR12;
	//format.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
	format.fmt.pix.width = width;
	format.fmt.pix.height = height;

	if (ioctl(fd, VIDIOC_S_FMT, &format) < 0){
		perror("Device could not set format, VIDIOC_S_FMT");
		return 1;
	}

	// Setup camera parameters
	v4l2_control ctl;
	ctl.id = V4L2_CID_GAIN;
	ctl.value = 32;
	ctl.id = V4L2_CID_EXPOSURE_ABSOLUTE;
	ctl.value = 1500;

	if (ioctl(fd, VIDIOC_S_CTRL, &ctl) < 0){
		perror("Device could not set controls, VIDIOC_S_CTRL");
		return 1;
	}

	// Request buffers from the device
	v4l2_requestbuffers bufrequest;
	bufrequest.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	bufrequest.memory = V4L2_MEMORY_MMAP;
	bufrequest.count = 1;

	if (ioctl(fd, VIDIOC_REQBUFS, &bufrequest) < 0){
		perror("Could not request buffer from device, VIDIOC_REQBUFS");
		return 1;
	}

	// Query the buffers
	v4l2_buffer bufquery = {0};
	bufquery.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	bufquery.memory = V4L2_MEMORY_MMAP;
	bufquery.index = 0;

	if(ioctl(fd, VIDIOC_QUERYBUF, &bufquery) < 0){
		perror("Device did not return the buffer information, VIDIOC_QUERYBUF");
		return 1;
	}

	char* buffer = (char*)mmap(
		NULL,
		bufquery.length,
		PROT_READ | PROT_WRITE,
		MAP_SHARED,
		fd,
		bufquery.m.offset);

	if (buffer == MAP_FAILED){
		perror("mmap");
		return 1;
	}

	memset(buffer, 0, bufquery.length);

	// Get a frame
	// Create a new buffer type
	v4l2_buffer bufferinfo;
	memset(&bufferinfo, 0, sizeof(bufferinfo));
	bufferinfo.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	bufferinfo.memory = V4L2_MEMORY_MMAP;
	bufferinfo.index = 0;

	// Activate streaming
	int type = bufferinfo.type;
	if (ioctl(fd, VIDIOC_STREAMON, &type) < 0){
		perror("Could not start streaming, VIDIOC_STREAMON");
		return 1;
	}

	Mat chs[2];
	int index = 1;
	char full_name[70];

	cout<<"Start Streaming"<<endl;
	
	// Initialize map for undistortion
	Mat mapx, mapy;
	cuda::GpuMat mapx_gpu, mapy_gpu;
	fisheye::initUndistortRectifyMap(::cameraMatrix, ::distCoeffs, cv::Matx33d::eye(), ::cameraMatrix, cv::Size(height, width), CV_32FC1, mapx, mapy);
	mapx_gpu.upload(mapx);
	mapy_gpu.upload(mapy);
	
	/***************************Loop***************************/
	while (true){
		// Queue the buffer
		if (ioctl(fd, VIDIOC_QBUF, &bufferinfo) < 0){
			perror("Could not queue buffer, VIDIOC_QBUF");
			return 1;

		}

		// Dequeue the buffer
		if (ioctl(fd, VIDIOC_DQBUF, &bufferinfo) < 0){
			perror("Could not queue buffer, VIDIOC_DQBUF");
			return 1;
		}

		//cout<<"convert buffer to Mat"<<endl;
		Mat frame_bayer_16bit(height, width, CV_16UC1, buffer);
		
		//cout<<"convert 16 bit to 8 bit"<<endl;
		frame_bayer_16bit_gpu.upload(frame_bayer_16bit);
		frame_bayer_16bit_gpu.convertTo(frame_bayer_8bit_gpu, CV_8UC1, 0.0605);

		//cout<<"convert BGGR to BGR"<<endl;
		cuda::cvtColor(frame_bayer_8bit_gpu, frame_bgr_8bit_gpu, COLOR_BayerBG2BGR);

		//cout<<"Undistort frame"<<endl;
		cuda::GpuMat frame_undistorted_gpu(height, width, CV_8UC3);
		cuda::remap(frame_distorted_gpu, frame_undistorted_gpu, mapx_gpu, mapy_gpu, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
		
		// Debug Section
		if (debug_frame == 1)
		{
			Mat frame_distorted;
			frame_bgr_8bit_gpu.download(frame_distorted);
			namedWindow("Fisheye", CV_WINDOW_NORMAL);
			resizeWindow("Fisheye", width/2, height/2);
			imshow("Fisheye", frame_distorted);
			waitKey(30);
		}

		/* Your Code here */
            

		/* End of Your Code */

		// press 'q' to end streaming
		int key = (waitKey(30) & 0xFF);
		if (key == 'q'){
			break;
		}
		// press 's' to save picture
		/*else if (key == 's'){
			//sprintf(full_name, "/home/xihui/vision/gps/calibration/calib_img_%d.jpg", index);
			sprintf(full_name, "/home/steven/Desktop/v4l2_camera/calibration/calib_img_%d.jpg", index);
			cout<<full_name<<endl;
			imwrite(full_name, frame_undistorted);
			index++;
		}
		else{}
		*/
	}

	/***************************Loop End***************************/

	//End stream
	if (ioctl(fd, VIDIOC_STREAMOFF, &type) < 0){
		perror("Could not end streaming, VIDIOC_STREAMOFF");
		return 1;
	}

	close(fd);
	cout<<"End Streaming"<<endl;
	return 0;
}
