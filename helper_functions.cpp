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

// This two functions are doing automatic white balance
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
