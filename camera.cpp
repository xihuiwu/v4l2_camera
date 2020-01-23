#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

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
 *     1936 x 1100
 */

int width = 1920;
int height = 1080;
/*
double K[9] = { 9.0325136355107918e+02, 0, 1.3755634776456659e+03, 0 ,
				9.0295296602633675e+02, 9.6835485779963028e+02,
				0, 0, 1};*/
double K[9] = { 278.8518, 0, 323.7384172, 0 ,
				273.17860196, 225.88590556,
				0, 0, 1};

//double D[4] = {-2.5271302020456785e-02, 6.1158058615096191e-03, -7.5600183621223269e-03, 1.8123937152019329e-03};
double D[4] = {-0.32653103,  0.08570291, -0.00523793, -0.00647632};

//last {-0.32653103,  0.08570291,  0.03115511, -0.00523793, -0.00647632}

Mat cameraMatrix = Mat(3, 3, CV_64FC1, &K);
Mat distCoeffs = Mat(4, 1, CV_64FC1, &D);


// Debug image type
string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

// white balance found online
void SimplestCB(Mat& in, Mat& out, float percent){
    assert(in.channels() == 3);
    assert(percent > 0 && percent < 100);

    float half_percent = percent / 200.0f;

    vector<Mat> tmpsplit; split(in,tmpsplit);
    for(int i=0;i<3;i++) {
        //find the low and high precentile values (based on the input percentile)
        Mat flat; tmpsplit[i].reshape(1,1).copyTo(flat);
        cv::sort(flat,flat,CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);
        int lowval = flat.at<uchar>(cvFloor(((float)flat.cols) * half_percent));
        int highval = flat.at<uchar>(cvCeil(((float)flat.cols) * (1.0 - half_percent)));
        //cout << lowval << " " << highval << endl;
        
        //saturate below the low percentile and above the high percentile
        tmpsplit[i].setTo(lowval,tmpsplit[i] < lowval);
        tmpsplit[i].setTo(highval,tmpsplit[i] > highval);
        
        //scale the channel
        normalize(tmpsplit[i],tmpsplit[i],0,255,NORM_MINMAX);
    }
    merge(tmpsplit,out);
}

void white_balance(Mat& src, Mat& dist){
	Mat chs[3], temp;

	cvtColor(src, temp, COLOR_BGR2Lab);
	split(temp, chs);
	Scalar avg_a = mean(chs[1]);
	Scalar avg_b = mean(chs[2]);
	chs[1] = chs[1] - ((avg_a.val[0] - 128)*(chs[0]/255)*1.1);
	chs[2] = chs[2] - ((avg_b.val[0] - 128)*(chs[0]/255)*1.1);
	vector<Mat> channels = {chs[0], chs[1], chs[2]};
	merge(channels, temp);
	cvtColor(temp, dist, COLOR_Lab2BGR);
}

int main(){



	cout<<"Initialization of Camera"<<endl;
	int fd;
	fd = open("/dev/video0", O_RDWR);
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
	ctl.value = 20;
	ctl.id = V4L2_CID_EXPOSURE_ABSOLUTE;
	ctl.value = 100;

	if (ioctl(fd, VIDIOC_S_CTRL, &ctl) < 0){
		perror("Device could not set controls, VIDIOC_S_CTRL");
		return 1;
	}
	
	// Set FPS
	struct v4l2_streamparm *setfps;
	setfps = (struct v4l2_streamparm *)calloc(1,sizeof(struct v4l2_streamparm));
	memset(setfps,0,sizeof(struct v4l2_streamparm));
	setfps->type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	setfps->parm.capture.timeperframe.numerator = 1;
	setfps->parm.capture.timeperframe.denominator = 30;
	if(ioctl(fd, VIDIOC_S_PARM,setfps) < 0)
	{
		perror("Could not set fps");
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
		Mat image_buffer(height, width, CV_16UC1);
		memcpy(image_buffer.data, &buffer[0], width*height*sizeof(uint16_t)); // only first channel will be copied to the Mat image_buffer

		//cout<<"convert 16 bit to 8 bit"<<endl;
		Mat frame_bayer_8bit(height, width, CV_8UC1);
		image_buffer.convertTo(frame_bayer_8bit, CV_8UC1);

		//cout<<"convert BGGR to BGR"<<endl;
		Mat frame_distorted(height, width, CV_8UC3);
		cvtColor(frame_bayer_8bit, frame_distorted, COLOR_BayerBG2BGR);

		//cout<<"Undistort frame"<<endl;
		//Mat frame_undistorted(height, width, CV_8UC3);
		//fisheye::undistortImage(frame_distorted, frame_undistorted, cameraMatrix, distCoeffs, cameraMatrix);

		namedWindow("Fisheye", CV_WINDOW_NORMAL);
		resizeWindow("Fisheye", width/2, height/2);
		imshow("Fisheye", frame_distorted);

		/* Your Code here */
            

		/* End of Your Code */

		int key = (waitKey(30) & 0xFF);

		// press 'q' to end streaming
		if (key == 'q'){
			break;
		}
		// press 's' to save picture
		else if (key == 's'){
			//sprintf(full_name, "/home/xihui/vision/gps/calibration/calib_img_%d.jpg", index);
			sprintf(full_name, "/home/steven/Desktop/v4l2_camera/calibration/calib_img_%d.jpg", index);
			cout<<full_name<<endl;
			imwrite(full_name, frame_distorted);
			index++;
		}
		else{
			
		}
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
