#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <map>

using namespace cv;
using namespace std;
using namespace cv::ml;

//比较Rect的高度的类函数
class Compare
{
public:
	int operator()(const Rect& x, const Rect& y) const
	{
		if (x.y >= y.y)
			return 0;
		return 1;
	}
};

//字符识别
int predict_number(Ptr<SVM> svm, Mat image, bool background) {

	Mat grayImageSrc;
	resize(image, image, Size(64, 64));
	cvtColor(image, grayImageSrc, COLOR_BGR2GRAY);
	//保证字符的灰度值比背景高
	if (!background)
		grayImageSrc = 255 - grayImageSrc;
	threshold(grayImageSrc, grayImageSrc, 180.0, 255.0, THRESH_BINARY | CV_THRESH_OTSU);

	//resize(image, image, Size(64, 64));
	HOGDescriptor *hog = new HOGDescriptor(cvSize(64, 64), cvSize(16, 16),
		cvSize(8, 8), cvSize(8, 8), 9);
	vector<float> descriptor;
	//计算hog特征
	hog->compute(grayImageSrc, descriptor, Size(2, 2), Size(0, 0));

	Mat testHog = Mat::zeros(1, descriptor.size(), CV_32FC1);
	int n = 0;
	for (vector<float>::iterator iter = descriptor.begin(); iter != descriptor.end(); iter++)
	{
		testHog.at<float>(0, n) = *iter;
		n++;
	}
	//svm识别
	int predictResult = svm->predict(testHog);

	return predictResult;
}

//这个函数没用到，用来显示各通道的统计图
void colorHist(Mat src, string histname)
{
	Mat hsv;
	vector<Mat> planes;

	cvtColor(src, hsv, CV_BGR2HSV);
	split(hsv, planes);

	int histSize = 256;

	float range[] = { 0, 255 };
	const float* histRange = { range };

	bool uniform = true; bool accumulate = false;
	Mat b_hist, g_hist, r_hist;

	/// Compute the histograms:
	calcHist(&planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

	imshow(histname + histname, planes[1]);

	// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	/// Normalize the result to [ 0, histImage.rows ]
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	/// Draw for each channel
	for (int i = 1; i < histSize; i++)
	{
		//line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
		//	Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
		//	Scalar(255, 0, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
			Scalar(0, 255, 0), 2, 8, 0);
		//line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
		//	Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
		//	Scalar(0, 0, 255), 2, 8, 0);
	}

	/// Display
	namedWindow(histname, CV_WINDOW_AUTOSIZE);
	imshow(histname, histImage);

	//waitKey(0);

}

//给水位划线的函数
float calcWaterLevel(Mat img, const float theta, const vector<Rect>& rects,const vector<int>& num_seq1, const float lenth_between) {
	
	if (num_seq1.size() < 2) return 0.0;

	//取出包含水位刻度区域
	float tl_x = rects[num_seq1.back()].tl().x - 2 * lenth_between;
	if (tl_x < 0) tl_x = 0;
	float tl_y = rects[num_seq1.back()].tl().y - 2 * lenth_between;
	if (tl_y < 0) tl_y = 0;
	float br_x = rects[num_seq1.back()].br().x + 2 * lenth_between;
	if (br_x > img.cols) br_x = img.cols;
	float br_y = rects[num_seq1.back()].br().y + 4 * lenth_between;
	if (br_y > img.rows) br_y = img.rows;

	Mat gray;
	Mat srcImage = img(Rect(tl_x, tl_y, br_x - tl_x, br_y - tl_y)).clone();

	//找到水平中心位置
	float mid_x = rects[num_seq1.back()].x + 0.5 * rects[num_seq1.back()].width - tl_x;
	vector<Vec2f> lines;
	float waterlevel = 0.0;
	//HoughLines(gray, lines, 1, CV_PI / 180, 30, 0, 0);

	Mat hsv;
	vector<Mat> planes;

	//转换到hsv空间
	cvtColor(srcImage, hsv, CV_BGR2HSV);
	split(hsv, planes);

	int l = max((int)mid_x - 3, 0);
	int r = min((int)mid_x + 3, hsv.cols);
	
	//取出中心及其相邻3像素的区域
	Mat slice = planes[1].colRange(l, r);
	//GaussianBlur(slice, slice, Size(5, 5), 3);
	blur(slice, slice, Size(5, 3));
	//colorHist(slice, "2");

	Mat strip;
	//水平方向累加
	reduce(slice, strip, 1, CV_REDUCE_AVG);
	
	//equalizeHist(strip, strip);
	//cout << strip << endl;
	
	int candidate = 0;

	
	bool blackedge = true;
	for (int i = strip.rows - 2; i > 0 ; i--)
	{	
		int cur = (int)strip.at<uchar>(i);
		int prev = (int)strip.at<uchar>(i - 1);	
		int last = (int)strip.at<uchar>(i + 1);
		//旋转黑边部分去除
		if (blackedge)
		{
			if (cur < 2)
			{
				continue;
			}
			if (abs(prev - cur) < 5)
			{
				blackedge = false;
				continue;
			}
		}
		//根据饱和度从下往上找到突变点，记为水线
		else
		{
			if (abs(prev - cur) > 15 || abs(last - prev) > 15)
			{
				candidate = i;
				break;
			}
		}
	}

	//cout << candidate << " : " << (int)strip.at<uchar>(candidate) << endl;

	Point pt1, pt2;
	pt1.x = mid_x - 100;
	pt1.y = candidate;
	pt2.x = mid_x + 100;
	pt2.y = candidate;
	line(srcImage, pt1, pt2, Scalar(55, 100, 195), 2, LINE_AA);


	//cout << strip << endl;
	//imshow("slice", slice);
	//waitKey(0);

	//*****
	//for (size_t i = 0; i < lines.size(); i++)
	//{
	//	float rho = lines[i][0], theta = lines[i][1];
	//	if (theta < (CV_PI / 180 * 30) || theta > (CV_PI / 180 * 150))
	//		continue; 

	//	Point pt1 = Point(0, 0), pt2 = Point(0, 0);
	//	double a = cos(theta), b = sin(theta);
	//	double x0 = a*rho, y0 = b*rho;
	//	pt1.x = cvRound(x0 + 1000 * (-b));
	//	pt1.y = cvRound(y0 + 1000 * (a));
	//	pt2.x = cvRound(x0 - 1000 * (-b));
	//	pt2.y = cvRound(y0 - 1000 * (a));

	//	//line(srcImage, pt1, pt2, Scalar(55,100,195), 1, LINE_AA);
	//	
	//	float tmp_y = y0 + (x0 - mid_x) * a;
	//	//if (tmp_y - rects[num_seq1.back()].y + tl_y > 2 * lenth_between /*|| tmp_y - rects[num_seq1.back()].y + tl_y < 0*/) continue;
	//	if (tmp_y > mid_y )
	//	{
	//		mid_y = tmp_y;
	//		max_pt1 = pt1;
	//		max_pt2 = pt2;
	//	}
	//}
	//line(srcImage, max_pt1, max_pt2, Scalar(55, 100, 255), 2, LINE_AA);
	imshow("src", srcImage);
	//waitKey(0);

	waterlevel = candidate + tl_y;
	return waterlevel;
}

//找到近似垂直的字符框区域，并求出与垂直线间的角度以及字符高度和面积均值
void find_vert_nums(const vector<Rect>& rects, vector<int>& num_seq, float& area, float& theta, float& lenth_between, bool first_time)
{
	map<int, vector<int>> up_downs;
	for (int i = 0; i < rects.size(); i++)
	{
		Rect up = rects[i];
		//cout << predict_number(svm, img(up)) << endl;
		int up_mid = up.x + up.width / 2;
		vector<int> downs;
		for (int j = i + 1; j < rects.size(); j++)
		{
			Rect down = rects[j];
			int down_mid = down.x + down.width / 2;
			if (down.y - (up.y + up.height) > 1.5 * up.height) break;
			if (first_time)
			{
				if (abs(up_mid - down_mid) > up.width ||
					(up.y + up.height) > down.y ||
					1.5 * up.area() < down.area() ||
					1.5 * down.area() < up.area())
					continue;
				downs.push_back(j);
			}
			else
			{
				if (abs(up_mid - down_mid) >0.5 * up.width ||
					(up.y + up.height) > down.y ||
					2 * up.area() < down.area() ||
					2 * down.area() < up.area())
					continue;
				downs.push_back(j);
			}
		}
		up_downs[i] = downs;
	}

	int length = rects.size();
	int max_start = length;
	int max_length = 0;
	vector<int> total_lenth(length);
	for (int i = length - 1; i >= 0; i--)
	{
		if (!up_downs.count(i))
			total_lenth[i] = 1;
		else
		{
			int max = 0;
			for (int j : up_downs[i])
			{
				if (total_lenth[j] > max) max = total_lenth[j];
			}
			total_lenth[i] = 1 + max;
		}
		if (total_lenth[i] > max_length)
		{
			max_length = total_lenth[i];
			max_start = i;
		}
	}

	for (int i = max_start, k = max_length; k > 0; k--) {
		num_seq.push_back(i);
		//rectangle(img_clone, rects[i], Scalar(0, 255, 0));
		if (!up_downs.count(i)) break;
		for (auto t : up_downs[i])
			if (total_lenth[t] == k - 1)
			{
				i = t;
				break;
			}
	}

	theta = 0;
	lenth_between = 0;
	area = rects[num_seq[0]].area();
	for (int i = 1; i < num_seq.size(); i++)
	{
		Rect prev = rects[num_seq[i - 1]];
		Rect cur = rects[num_seq[i]];
		float prev_mid_x = (float)prev.x + prev.width / 2;
		float prev_mid_y = (float)prev.y + prev.height / 2;
		float cur_mid_x = (float)cur.x + cur.width / 2;
		float cur_mid_y = (float)cur.y + cur.height / 2;

		area += cur.area();
		lenth_between += sqrt((cur_mid_y - prev_mid_y) * (cur_mid_y - prev_mid_y) + (cur_mid_x - prev_mid_x) * (cur_mid_x - prev_mid_x));
		theta += atan2f(cur_mid_y - prev_mid_y, cur_mid_x - prev_mid_x) * 180 / CV_PI;
	}

	area = area / num_seq.size();
	theta = theta / (num_seq.size() - 1);
	lenth_between = lenth_between / (num_seq.size() - 1) / 2;
}

float detect_number(Mat& img, Ptr<SVM> svm, int current_frame, float prevnum) {
	Mat img_clone = img.clone();
	if (img.empty()) {
		cout << "img is null" << endl;
	}
	
	//colorHist(img, "1");

	//before rotation
	Mat gray_img;
	cvtColor(img_clone, gray_img, COLOR_BGR2GRAY);
	GaussianBlur(gray_img, gray_img, Size(3, 3), 1);
	Canny(gray_img, gray_img, 200, 255);

	//namedWindow("gray_img_binary", WINDOW_NORMAL);
	//imshow("gray_img_binary", gray_img);
	//waitKey(0);

	//找到字符连通区域
	vector<vector<Point> > contours;
	vector<Point> convex;
	vector<Vec4i> hierarchy;
	findContours(gray_img, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	vector<Rect> rects;
	for (int j = 0; j < contours.size(); j++) {
		cv::Rect rect = cv::boundingRect(contours[j]);
		float wh = (float)rect.width / rect.height;
		if (wh < 0.5 || wh > 1.5)
			continue;

		//rectangle(img_clone, rect, Scalar(255, 0, 0));
		rects.push_back(rect);
	}
	
	if (rects.size() <= 2) return prevnum;

	//根据字符框水平坐标排序
	sort(rects.begin(), rects.end(),
		bind(less<size_t>(),
			bind(&Rect::y, std::placeholders::_1),
			bind(&Rect::y, std::placeholders::_2)
		)
	);

	float area, theta, lenth_between;
	vector<int> num_seq;
	//找到包含字符框最多的直线
	find_vert_nums(rects, num_seq, area, theta, lenth_between, true);
	 
	//imshow("img", img_clone);
	//waitKey(0);

	//根据得到的角度旋转
	Point2f center = Point2f(img.cols / 2.0, img.rows / 2.0);
	Mat R = getRotationMatrix2D(center, theta - 90, 1);
	Mat rotate;
	//warpAffine(img, rotate, R, img.size());

	Rect bbox = cv::RotatedRect(center, img.size(), theta - 90).boundingRect();

	R.at<double>(0, 2) += bbox.width / 2.0 - center.x;
	R.at<double>(1, 2) += bbox.height / 2.0 - center.y;

	warpAffine(img, rotate, R, bbox.size());

	//imshow("rotate", rotate);
	//waitKey(0);

	//对旋转后的图像进行二值化
	//after rotation
	Mat gray1;
	Mat gray_img1;
	cvtColor(rotate, gray1, COLOR_BGR2GRAY);
	//GaussianBlur(gray_img1, gray_img1, Size(5, 5), 1);
	threshold(gray1, gray_img1, 180, 255, CV_THRESH_BINARY);
	morphologyEx(gray_img1, gray_img1, MORPH_DILATE, getStructuringElement(MORPH_RECT, Size(2, 3)));
	//Canny(gray_img1, gray_img1, 200, 255);

	//imshow("gray", gray_img1);
	//waitKey(0);

	//重新寻找字符连通域
	vector<vector<Point> > contours1;
	vector<Point> convex1;
	vector<Vec4i> hierarchy1;
	findContours(gray_img1, contours1, hierarchy1, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

	//处理包含两种颜色字符的情况
	vector<vector<Point> > contours2;
	if (contours1.size() < contours.size()/2) 
	{
		Mat gray_img2 = 255 - gray1;

		threshold(gray_img2, gray_img2, 160, 255, CV_THRESH_BINARY);
		morphologyEx(gray_img2, gray_img2, MORPH_DILATE, getStructuringElement(MORPH_RECT, Size(2, 3)));
		//imshow("gray_img2", gray_img2);
		//waitKey(0);
		vector<Point> convex2;
		vector<Vec4i> hierarchy2;
		findContours(gray_img2, contours2, hierarchy2, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
	}

	Mat rotate1 = rotate.clone();

	vector<Rect> rects1;

	map<Rect, bool, Compare> color_map;
	//背景为深色时的字符
	for (int j = 0; j < contours1.size(); j++) {
		cv::Rect rect = cv::boundingRect(contours1[j]);
		float wh = (float)rect.width / rect.height;
		if (wh < 0.1 || wh > 1.5 || rect.area() < area * 0.2 || rect.area() > area * 5)
			continue;

		color_map.insert(pair<Rect, bool>(rect, true));
		//rectangle(rotate1, rect, Scalar(255, 0, 0));
		rects1.push_back(rect);
	}

	//背景为深色时的字符
	for (int k = 0; k < contours2.size(); k++) {
		cv::Rect rect = cv::boundingRect(contours2[k]);
		float wh = (float)rect.width / rect.height;
		if (wh < 0.1 || wh > 1.5 || rect.area() < area * 0.2 || rect.area() > area * 5)
			continue;

		color_map.insert(pair<Rect, bool>(rect, false));
		//rectangle(rotate1, rect, Scalar(0, 255, 0));
		rects1.push_back(rect);
	}

	//cout << rects1.size() << endl;
	//imshow("img", rotate1);
	//waitKey(0);
	if (rects1.size() <= 2) return prevnum;

	sort(rects1.begin(), rects1.end(),
		bind(less<size_t>(),
			bind(&Rect::y, std::placeholders::_1),
			bind(&Rect::y, std::placeholders::_2)
		)
	);

	//area : the average area of every letter or number
	//theta : the angle of water ruler to horizontal
	//lenth_between(typo) : average height of every letter or number
	float area1, theta1, lenth_between1;
	vector<int> num_seq1;
	//重新找字符区域，这时候把约束增强，去除伪字符
	find_vert_nums(rects1, num_seq1, area1, theta1, lenth_between1, false);

	//for (int k = 0; k < num_seq1.size(); k++) {
	//	rectangle(rotate1, rects1[num_seq1[k]], Scalar(0, 255, 0));
	//}

	//cout << rects1.size() << endl;
	//imshow("img", rotate1);
	//waitKey(0);

	//计算出水线的位置
	float waterlevel = calcWaterLevel(rotate, theta1, rects1, num_seq1, lenth_between1);

	map<float, int> predict_last;
	float last_num = 0.0;
	for (int i = 0; i < num_seq1.size(); i++)
	{
		int k = num_seq1[i];
		//imshow("num", img(rects1[k]));
		//waitKey(0);
		//如果检测到M字符，处理其前部字符
		if (predict_number(svm, rotate(rects1[k]), color_map[rects1[k]]) == 10)
		{
			int num2 = 0;
			Rect num2_rect;
			for (Rect r : rects1)
			{
				if (rects1[k].x > r.x + r.width && rects1[k].x - 1 * rects1[k].width < r.x + r.width
					&& rects1[k].y < r.y + 1.0 * r.height && rects1[k].y + rects1[k].height > r.y + 0.5 * r.height)
				{
					num2 = predict_number(svm, rotate(r), color_map[r]);
					//imshow("num2", rotate(r));
					//waitKey(0);
					//cout << "num2 : " << num2 << endl;
					num2_rect = r;
					break;
				}
			}
			//如果识别出前部字符，继续处理其之前字符
			if (num2_rect.x == 0 || num2 < 0 || num2 > 9) continue;
			int num1 = 0;
			for (Rect r : rects1)
			{
				if (num2_rect.x > r.x + r.width && num2_rect.x - 1 * num2_rect.width < r.x + r.width
					&& num2_rect.y < r.y + 0.5 * r.height && num2_rect.y + num2_rect.height > r.y + 0.5 * r.height)
				{
					num1 = predict_number(svm, rotate(r), color_map[r]);
					//imshow("num1", rotate(r));
					//cout << "num1 : " << num1 << endl;
					//waitKey(0);
					break;
				}
			}

			if (num1 < 0 || num1 > 9) continue;
			int scale = 0;
			//计算字符水位
			scale = num1 * 10 + num2;
			//cout << scale << endl;
			last_num = scale - (num_seq1.size() - i - 1) * 0.2;
			if (predict_last.count(last_num)) predict_last[last_num] ++;
			else predict_last[last_num] = 1;
		}
	}
	
	//找到最低的水位刻度
	float max_time_num;
	int max_time = 0;
	for (auto it : predict_last)
	{
		if (it.second > max_time || it.first > max_time_num)
		{
			max_time = it.second;
			max_time_num = it.first;
		}
	}
	
	//通过距离判断当前水位高度
	if (!max_time) return prevnum;
	else 
	{
		float diff = (waterlevel - rects1[num_seq1.back()].tl().y) / lenth_between1 * 0.1;
		float curnum = max_time_num + 0.1 - diff;
		//if (abs(curnum - prevnum) > 0.2 && prevnum > 0.1) return prevnum;
		return curnum;
	}
}

int main() {
	//载入svm模型
	Ptr<SVM> svm = Algorithm::load<SVM>("9_25_svm.xml");

	Mat frame;
	//载入视频流
	VideoCapture capture("D:\\waterlevel\\new_video\\3.mov");
	float prevnum = 0.0f, curnum  = 0.0f;
	while(1)
	{
		capture >> frame;
		int frame_num = capture.get(CV_CAP_PROP_POS_FRAMES);
		cout << "current frame is :" << frame_num << endl;

		if (frame.empty())         
			break;
		//检测水位数字
		curnum = detect_number(frame, svm, frame_num, prevnum);
		stringstream stream;
		stream << fixed << setprecision(2) << curnum;
		string result_str = stream.str();
		
		putText(frame, result_str, Point(50, 80), FONT_HERSHEY_PLAIN, 2.0, CV_RGB(0, 255, 0), 2.0);
		imshow("result", frame);
		waitKey(20);
		prevnum = curnum;
	}
	capture.release();
	return 0;
}