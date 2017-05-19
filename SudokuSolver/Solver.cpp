// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <queue>
#include <time.h>

#define MIN_AREA_THRESHOLD 20

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/*
This binarization separates the big window into 4 different smaller pictures and binarizes them separately
*/
Mat blockBinarization(Mat src, double thresholdMultiplication){
	Mat binary(src.size(), CV_8UC1);

	int nrows = src.rows;
	int ncols = src.cols;

	// calculate 4 different average values for the 4 sections of the picture
	int medCols = ncols / 2;
	int medRows = nrows / 2;
	int count = ncols * nrows / 4;
	int avgTL, avgTR, avgBL, avgBR;

	// Top left
	avgTL = 0;
	for (int i = 0; i < medRows; i++)
		for (int j = 0; j < medCols; j++)
			avgTL += src.at<uchar>(i,j);
	// Top Right
	avgTR = 0;
	for (int i = 0; i < medRows; i++)
		for (int j = medCols; j < ncols; j++)
			avgTR += src.at<uchar>(i, j);
	// Bottom left
	avgBL = 0;
	for (int i = medRows; i < nrows; i++)
		for (int j = 0; j < medCols; j++)
			avgBL += src.at<uchar>(i, j);
	// Top left
	avgBR = 0;
	for (int i = medRows; i < nrows; i++)
		for (int j = medCols; j < ncols; j++)
			avgBR += src.at<uchar>(i, j);

	avgTL = avgTL * thresholdMultiplication / count;
	avgTR = avgTR * thresholdMultiplication / count;
	avgBL = avgBL * thresholdMultiplication / count;
	avgBR = avgBR * thresholdMultiplication / count;

	// color the pixels of the destination image
	// Top left
	for (int i = 0; i < medRows; i++)
		for (int j = 0; j < medCols; j++)
			binary.at<uchar>(i, j) = src.at<uchar>(i, j) > avgTL ? 0 : 255;
	// Top Right
	for (int i = 0; i < medRows; i++)
		for (int j = medCols; j < ncols; j++)
			binary.at<uchar>(i, j) = src.at<uchar>(i, j) > avgTR ? 0 : 255;
	// Bottom left
	for (int i = medRows; i < nrows; i++)
		for (int j = 0; j < medCols; j++)
			binary.at<uchar>(i, j) = src.at<uchar>(i, j) > avgBL ? 0 : 255;
	// Top left
	for (int i = medRows; i < nrows; i++)
		for (int j = medCols; j < ncols; j++)
			binary.at<uchar>(i, j) = src.at<uchar>(i, j) > avgBR ? 0 : 255;

	return binary;
}

/*
The function labels all distinct elements from src image.
If keep == true then all elements are removed except the largest CC
If keep == false then the smaller elements are kept.
*/
Mat extractLargestCC(Mat src, bool keepLargest){
	Mat dst(src.size(), CV_8UC1);

	int nrows = src.rows;
	int ncols = src.cols;

	int label = 0; // first label will be 1, 0 is not used
	std::vector<int> areas;
	// generate the labels matrix dynamically
	int** labels = new int*[nrows];
	for (int i = 0; i < nrows; i++)
		labels[i] = new int[ncols];

	// initialize all labels to 0
	for (int i = 0; i < nrows; i++)
		for (int j = 0; j < ncols; j++)
			labels[i][j] = 0;

	for (int i = 0; i < nrows; i++)
		for (int j = 0; j < ncols; j++){
			if (src.at<uchar>(i, j) == 255 && labels[i][j] == 0){
				label++;
				std::queue<Point2i> Q = std::queue<Point2i>();
				labels[i][j] = label;
				Q.push(Point2i(i, j));

				// calculate the area of the component
				int area = 1;

				while (Q.size()){
					Point2i q = Q.front(); Q.pop();
					// for all 8 neighbors
					for (int m = q.x - 1; m <= q.x + 1; m++)
						for (int n = q.y - 1; n <= q.y + 1; n++){
							// assure that no invalid element is accessed
							if (m >= 0 && n >= 0 && m < nrows && n < ncols){
								if (src.at<uchar>(m, n) == 255 && labels[m][n] == 0){
									labels[m][n] = label;
									Q.push(Point2i(m, n));
									area++;
								}
							}
						}
				}
				areas.push_back(area);
			}
		}

	int largestArea = 0;
	int largestAreaLabel = 0;
	for (int i = 0; i < areas.size(); i++){
		if (areas.at(i) > largestArea){
			largestArea = areas.at(i);
			largestAreaLabel = i + 1;
		}
	}

	
	for (int i = 0; i < nrows; i++){
		for (int j = 0; j < ncols; j++){
			if (keepLargest){
				// color only the largest component with white, the others will be background
				if (labels[i][j] == largestAreaLabel){
					dst.at<uchar>(i, j) = 255;
				}
				else {
					dst.at<uchar>(i, j) = 0; // black color
				}
			}
			else {
				// color only the small components with white, the largest will become background
				if (labels[i][j] != largestAreaLabel){
					dst.at<uchar>(i, j) = labels[i][j] != 0 ? 255 : 0;
				}
				else {
					dst.at<uchar>(i, j) = 0; // almost black color
				}
			}
		}
		
	}


	// free the allocated memory for labels matrix
	for (int i = 0; i < nrows; i++)
		delete[]labels[i];
	delete[]labels;

	return dst;
}

void drawLine(Vec2f line, Mat &img, Scalar rgb = CV_RGB(0, 0, 255)) { 
	if (line[1] != 0) { 
		float m = -1 / tan(line[1]); 
		float c = line[0] / sin(line[1]); 
		cv::line(img, Point(0, c), Point(img.size().width, m*img.size().width + c), rgb); 
	} 
	else { 
		cv::line(img, Point(line[0], 0), Point(line[0], img.size().height), rgb); 
	} 
} 

void mergeLines(vector<Vec2f> *lines, Mat &src){
	
	vector<Vec2f>::iterator current;
	for (current = lines->begin(); current != lines->end(); current++){
		if ((*current)[0] == 0 && (*current)[1] == -100) continue;

		float p1 = (*current)[0];
		float theta1 = (*current)[1];

		Point pt1current, pt2current; 
		if (theta1>CV_PI * 45 / 180 && theta1<CV_PI * 135 / 180) { 
			pt1current.x = 0; 
			pt1current.y = p1 / sin(theta1); 
			pt2current.x = src.size().width; 
			pt2current.y = -pt2current.x / tan(theta1) + p1 / sin(theta1); 
		}
		else { 
			pt1current.y = 0; 
			pt1current.x = p1 / cos(theta1); 
			pt2current.y = src.size().height; 
			pt2current.x = -pt2current.y / tan(theta1) + p1 / cos(theta1); 
		} 

		vector<Vec2f>::iterator    pos; 
		for (pos = lines->begin(); pos != lines->end(); pos++) {
			if (*current == *pos) continue; 

			if (fabs((*pos)[0] - (*current)[0])<20 && fabs((*pos)[1] - (*current)[1])<CV_PI * 10 / 180) {
				float p = (*pos)[0]; 
				float theta = (*pos)[1];

				Point pt1, pt2; 
				if ((*pos)[1]>CV_PI * 45 / 180 && (*pos)[1]<CV_PI * 135 / 180) { 
					pt1.x = 0; pt1.y = p / sin(theta); 
					pt2.x = src.size().width; 
					pt2.y = -pt2.x / tan(theta) + p / sin(theta); 
				}
				else { 
					pt1.y = 0; pt1.x = p / cos(theta); 
					pt2.y = src.size().height;
					pt2.x = -pt2.y / tan(theta) + p / cos(theta); 
				} 

				if (((double)(pt1.x - pt1current.x)*(pt1.x - pt1current.x) + (pt1.y - pt1current.y)*(pt1.y - pt1current.y)<64 * 64) 
					&& ((double)(pt2.x - pt2current.x)*(pt2.x - pt2current.x) + (pt2.y - pt2current.y)*(pt2.y - pt2current.y)<64 * 64)) { 
					// Merge the two 
					(*current)[0] = ((*current)[0]+(*pos)[0])/2; 
					(*current)[1] = ((*current)[1]+(*pos)[1])/2; 
					(*pos)[0]=0; (*pos)[1]=-100;
				} 
			}
		}
	}
}

vector<Vec2f> detectMarginalLines(vector<Vec2f> lines){
	Vec2f topEdge = Vec2f(1000, 1000);
	double topYIntercept = 100000, topXIntercept = 0;
	Vec2f bottomEdge = Vec2f(-1000, -1000);
	double bottomYIntercept = 0, bottomXIntercept = 0;
	Vec2f leftEdge = Vec2f(1000, 1000);
	double leftXIntercept = 100000, leftYIntercept = 0;
	Vec2f rightEdge = Vec2f(-1000, -1000);
	double rightXIntercept = 0, rightYIntercept = 0;

	for (int i = 0; i<lines.size(); i++)     {
		Vec2f current = lines[i];
		float p = current[0];
		float theta = current[1];

		if (p == 0 && theta == -100)
			continue;

		double xIntercept, yIntercept;
		xIntercept = p / cos(theta);
		yIntercept = p / (cos(theta)*sin(theta));

		if (theta>CV_PI * 80 / 180 && theta<CV_PI * 100 / 180){
			if (p<topEdge[0])
				topEdge = current;
			if (p>bottomEdge[0])
				bottomEdge = current;
		}
		else if (theta<CV_PI * 10 / 180 || theta>CV_PI * 170 / 180){
			if (xIntercept>rightXIntercept) {
				rightEdge = current;
				rightXIntercept = xIntercept;
			}
			else if (xIntercept <= leftXIntercept) {
				leftEdge = current;
				leftXIntercept = xIntercept;
			}
		}
	}

	vector<Vec2f> result;
	result.push_back(topEdge);
	result.push_back(bottomEdge);
	result.push_back(leftEdge);
	result.push_back(rightEdge);

	return result;
}

vector<CvPoint> detectIntersectionPoints(Vec2f topEdge, Vec2f bottomEdge, Vec2f leftEdge, Vec2f rightEdge, Mat src){
	Point left1, left2, right1, right2, bottom1, bottom2, top1, top2;
	int height = src.size().height;
	int width = src.size().width;

	// find 2 points on a line
	if (leftEdge[1] != 0) {
		left1.x = 0;
		left1.y = leftEdge[0] / sin(leftEdge[1]);
		left2.x = width;
		left2.y = -left2.x / tan(leftEdge[1]) + left1.y;
	}
	else{
		left1.y = 0;
		left1.x = leftEdge[0] / cos(leftEdge[1]);
		left2.y = height;
		left2.x = left1.x - height*tan(leftEdge[1]);
	}
	if (rightEdge[1] != 0) {
		right1.x = 0;
		right1.y = rightEdge[0] / sin(rightEdge[1]);
		right2.x = width;
		right2.y = -right2.x / tan(rightEdge[1]) + right1.y;
	}
	else {
		right1.y = 0;
		right1.x = rightEdge[0] / cos(rightEdge[1]);
		right2.y = height;
		right2.x = right1.x - height*tan(rightEdge[1]);
	}
	bottom1.x = 0;
	bottom1.y = bottomEdge[0] / sin(bottomEdge[1]);
	bottom2.x = width; bottom2.y = -bottom2.x / tan(bottomEdge[1]) + bottom1.y;
	top1.x = 0;
	top1.y = topEdge[0] / sin(topEdge[1]);
	top2.x = width;
	top2.y = -top2.x / tan(topEdge[1]) + top1.y;

	// calculate the actual intersection
	double leftA = left2.y - left1.y;
	double leftB = left1.x - left2.x;
	double leftC = leftA*left1.x + leftB*left1.y;
	double rightA = right2.y - right1.y;
	double rightB = right1.x - right2.x;
	double rightC = rightA*right1.x + rightB*right1.y;
	double topA = top2.y - top1.y;
	double topB = top1.x - top2.x;
	double topC = topA*top1.x + topB*top1.y;
	double bottomA = bottom2.y - bottom1.y;
	double bottomB = bottom1.x - bottom2.x;
	double bottomC = bottomA*bottom1.x + bottomB*bottom1.y;

	vector<CvPoint> points;

	// Intersection of left and top    
	double detTopLeft = leftA*topB - leftB*topA;
	points.push_back(cvPoint((topB*leftC - leftB*topC) / detTopLeft, (leftA*topC - topA*leftC) / detTopLeft));
	// Intersection of top and right    
	double detTopRight = rightA*topB - rightB*topA;
	points.push_back(cvPoint((topB*rightC - rightB*topC) / detTopRight, (rightA*topC - topA*rightC) / detTopRight));
	// Intersection of right and bottom   
	double detBottomRight = rightA*bottomB - rightB*bottomA;
	points.push_back(cvPoint((bottomB*rightC - rightB*bottomC) / detBottomRight, (rightA*bottomC - bottomA*rightC) / detBottomRight));
	// Intersection of bottom and left    
	double detBottomLeft = leftA*bottomB - leftB*bottomA;
	points.push_back(cvPoint((bottomB*leftC - leftB*bottomC) / detBottomLeft, (leftA*bottomC - bottomA*leftC) / detBottomLeft));

	return points;
}

int getObjectMinX(Mat src){
	for (int i = 0; i < src.rows; i++){
		for (int j = 0; j < src.cols; j++){
			if (src.at<uchar>(i, j) != 0)
				return i;
		}
	}
	return 0;
}

int getObjectMinY(Mat src){
	for (int j = 0; j < src.cols; j++){
		for (int i = 0; i < src.rows; i++){
			if (src.at<uchar>(i, j) != 0)
				return j;
		}
	}
	return 0;
}

Point2i getObjectMin(Mat src){
	return Point2i(getObjectMinY(src), getObjectMinX(src));
}

int getObjectMaxX(Mat src){
	for (int i = src.rows - 1; i >= 0; i--){
		for (int j = src.cols - 1; j >= 0; j--){
			if (src.at<uchar>(i, j) != 0)
				return i; //+ MARGIN < src.rows ? i + MARGIN : src.rows - 1;
		}
	}
	return 0;
}

int getObjectMaxY(Mat src){
	for (int j = src.cols - 1; j >= 0; j--){
		for (int i = src.rows - 1; i >= 0; i--){
			if (src.at<uchar>(i, j) != 0)
				return j; //+ MARGIN < src.cols ? j + MARGIN : src.cols - 1;
		}
	}
	return 0;
}

Point2i getObjectMax(Mat src){
	return Point2i(getObjectMaxY(src), getObjectMaxX(src));
}

Mat negative(Mat src){
	Mat dst(src.size(), CV_8UC1);
	for (int i = 0; i < src.rows; i++){
		for (int j = 0; j < src.cols; j++){
			dst.at<uchar>(i, j) = src.at<uchar>(i, j) == 0 ? 255 : 0;
		}
	}
	return dst;
}

Mat imageContour(Mat src, int objectColor){
	Mat dst(src.size(), CV_8UC1);
	int nrows = src.rows;
	int ncols = src.cols;
	int background = objectColor == 0 ? 255 : 0;

	// fill with background
	for (int i = 0; i < nrows; i++)
		for (int j = 0; j < ncols; j++){
			dst.at<uchar>(i, j) = background;
		}
	

	for (int r = 1; r < nrows - 1; r++){
		for (int c = 1; c < ncols - 1; c++){

			if (src.at<uchar>(r,c) == objectColor){
				bool perimeterPoint = false;
				for (int i = r - 1; i <= r + 1; i++)
					for (int j = c - 1; j <= c + 1; j++){
						if (src.at<uchar>(i,j) != objectColor){
							perimeterPoint = true;
						}
					}
				// verify if is perimeter point
				if (perimeterPoint){
					dst.at<uchar>(r, c) = objectColor;
				}
			}
		}
	}

	return dst;
}

float patternMatchingScore(Mat dt, Mat unknown, int objectColor){
	float sum = 0.0;
	int count = 0;

	for (int i = 0; i < unknown.rows; i++)
		for (int j = 0; j < unknown.cols; j++){
			if (unknown.at<uchar>(i, j) == objectColor){
				sum += dt.at<float>(i, j);
				count++;
			}
		}
	
	if (count > 89) 
		return sum / count;
	else 
		return 10.0;
}

void thinningIteration(cv::Mat& img, int iter)
{
	CV_Assert(img.channels() == 1);
	CV_Assert(img.depth() != sizeof(uchar));
	CV_Assert(img.rows > 3 && img.cols > 3);

	cv::Mat marker = cv::Mat::zeros(img.size(), CV_8UC1);

	int nRows = img.rows;
	int nCols = img.cols;

	if (img.isContinuous()) {
		nCols *= nRows;
		nRows = 1;
	}

	int x, y;
	uchar *pAbove;
	uchar *pCurr;
	uchar *pBelow;
	uchar *nw, *no, *ne;    // north (pAbove)
	uchar *we, *me, *ea;
	uchar *sw, *so, *se;    // south (pBelow)

	uchar *pDst;

	// initialize row pointers
	pAbove = NULL;
	pCurr = img.ptr<uchar>(0);
	pBelow = img.ptr<uchar>(1);

	for (y = 1; y < img.rows - 1; ++y) {
		// shift the rows up by one
		pAbove = pCurr;
		pCurr = pBelow;
		pBelow = img.ptr<uchar>(y + 1);

		pDst = marker.ptr<uchar>(y);

		// initialize col pointers
		no = &(pAbove[0]);
		ne = &(pAbove[1]);
		me = &(pCurr[0]);
		ea = &(pCurr[1]);
		so = &(pBelow[0]);
		se = &(pBelow[1]);

		for (x = 1; x < img.cols - 1; ++x) {
			// shift col pointers left by one (scan left to right)
			nw = no;
			no = ne;
			ne = &(pAbove[x + 1]);
			we = me;
			me = ea;
			ea = &(pCurr[x + 1]);
			sw = so;
			so = se;
			se = &(pBelow[x + 1]);

			int A = (*no == 0 && *ne == 1) + (*ne == 0 && *ea == 1) +
				(*ea == 0 && *se == 1) + (*se == 0 && *so == 1) +
				(*so == 0 && *sw == 1) + (*sw == 0 && *we == 1) +
				(*we == 0 && *nw == 1) + (*nw == 0 && *no == 1);
			int B = *no + *ne + *ea + *se + *so + *sw + *we + *nw;
			int m1 = iter == 0 ? (*no * *ea * *so) : (*no * *ea * *we);
			int m2 = iter == 0 ? (*ea * *so * *we) : (*no * *so * *we);

			if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
				pDst[x] = 1;
		}
	}

	img &= ~marker;
}

void thinning(const cv::Mat& src, cv::Mat& dst)
{
	dst = src.clone();
	dst /= 255;         // convert to binary image

	cv::Mat prev = cv::Mat::zeros(dst.size(), CV_8UC1);
	cv::Mat diff;

	do {
		thinningIteration(dst, 0);
		thinningIteration(dst, 1);
		cv::absdiff(dst, prev, diff);
		dst.copyTo(prev);
	} while (cv::countNonZero(diff) > 0);

	dst *= 255;
}


void startSolver(){
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		///////////////////////////////////////////
		// Step 0 - load image
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		
		// create structural element for morphological operations
		uchar kernelData[] = { 0, 1, 0, 1, 1, 1, 0, 1, 0 };
		Mat kernel(3, 3, CV_8UC1, kernelData);

		///////////////////////////////////////////
		// Step 1 - binarize 
		Mat binarized = blockBinarization(src, (double)1/2); 
		dilate(binarized, binarized, kernel); 

		imshow("binarized", binarized);
				
		///////////////////////////////////////////
		// Step 2 - extract largest connected component
		Mat largestCC = extractLargestCC(binarized, true);

		//imshow("largest CC", largestCC);

		///////////////////////////////////////////
		// Step 3 - Hough transform line detection
		vector<Vec2f> lines; 
		
		HoughLines(largestCC, lines, 1, CV_PI / 180, 200); 

		mergeLines(&lines, largestCC);

		for (int i = 0; i<lines.size(); i++) { 
			drawLine(lines[i], largestCC, CV_RGB(0, 0, 128)); 
		} 

		//imshow("Hough tranform", largestCC);

		///////////////////////////////////////////
		// Step 4.1 - detect marginal lines
		vector<Vec2f> marginalLines = detectMarginalLines(lines);
		Vec2f topEdge = marginalLines.at(0);
		Vec2f bottomEdge = marginalLines.at(1);
		Vec2f leftEdge = marginalLines.at(2);
		Vec2f rightEdge = marginalLines.at(3);

		//draw the marginal lines
		//drawLine(topEdge, src, CV_RGB(0, 0, 0));   
		//drawLine(bottomEdge, src, CV_RGB(0, 0, 0));  
		//drawLine(leftEdge, src, CV_RGB(0, 0, 0));   
		//drawLine(rightEdge, src, CV_RGB(0, 0, 0)); 

		///////////////////////////////////////////
		// Step 4.2 - detect intersection point of marginal lines
		vector<CvPoint> intersectionPoints = detectIntersectionPoints(topEdge, bottomEdge, leftEdge, rightEdge, src);
		CvPoint ptTopLeft = intersectionPoints.at(0);
		CvPoint ptTopRight = intersectionPoints.at(1);
		CvPoint ptBottomRight = intersectionPoints.at(2);
		CvPoint ptBottomLeft = intersectionPoints.at(3);

		// after finding the points, we can extract the puzzle box from the image based on the 4 marginal points
		int maxLength = (ptBottomLeft.x - ptBottomRight.x)*(ptBottomLeft.x - ptBottomRight.x) + (ptBottomLeft.y - ptBottomRight.y)*(ptBottomLeft.y - ptBottomRight.y);  
		int temp = (ptTopRight.x - ptBottomRight.x)*(ptTopRight.x - ptBottomRight.x) + (ptTopRight.y - ptBottomRight.y)*(ptTopRight.y - ptBottomRight.y);   
		
		if (temp>maxLength) 
			maxLength = temp;   
		temp = (ptTopRight.x - ptTopLeft.x)*(ptTopRight.x - ptTopLeft.x) + (ptTopRight.y - ptTopLeft.y)*(ptTopRight.y - ptTopLeft.y);    
		
		if (temp>maxLength)
			maxLength = temp;   
		temp = (ptBottomLeft.x - ptTopLeft.x)*(ptBottomLeft.x - ptTopLeft.x) + (ptBottomLeft.y - ptTopLeft.y)*(ptBottomLeft.y - ptTopLeft.y);  
		
		if (temp>maxLength) 
			maxLength = temp;    
		maxLength = sqrt((double)maxLength);

		// create destination points
		Point2f srcP[4], dstP[4]; 
		srcP[0] = ptTopLeft;           
		dstP[0] = Point2f(0, 0); 
		srcP[1] = ptTopRight;      
		dstP[1] = Point2f(maxLength - 1, 0);
		srcP[2] = ptBottomRight;  
		dstP[2] = Point2f(maxLength - 1, maxLength - 1);
		srcP[3] = ptBottomLeft;   
		dstP[3] = Point2f(0, maxLength - 1); 

		Mat puzzleMatrix = Mat(Size(maxLength, maxLength), CV_8UC1);
		warpPerspective(src, puzzleMatrix, getPerspectiveTransform(srcP, dstP), Size(maxLength, maxLength));

		imshow("Puzzle", puzzleMatrix);

		///////////////////////////////////////////
		// Step 5 - after finding the puzzle box, we can start extracting the digits from the elementar boxes

		// binarize the image
		//Mat binarizedPuzzle = binarize(puzzleMatrix, averagePixelValue(puzzleMatrix)*2/3);
		Mat binarizedPuzzle = blockBinarization(puzzleMatrix, (double)2/3);

		// remove the grid for easier template matching
		binarizedPuzzle = extractLargestCC(binarizedPuzzle, false);
		dilate(binarizedPuzzle, binarizedPuzzle, kernel);

		imshow("Binarized puzzle", binarizedPuzzle);

		// create the elementar box
		int boxSize = ceil((double)maxLength / 9.0);
		
		Mat** boxes = new Mat*[9];
		for (int i = 0; i < 9; i++)
			boxes[i] = new Mat[9];
		

		// destination points will be always the same
		Point2f toP[4];
		toP[0] = Point2f(0, 0);
		toP[1] = Point2f(0, boxSize - 1);
		toP[2] = Point2f(boxSize - 1, boxSize - 1);
		toP[3] = Point2f(boxSize - 1, 0);

		// iterate through boxes and create an elementar cell
		for (int i = 0; i < 9; i++){
			for (int j = 0; j < 9; j++){
				boxes[i][j] = Mat(boxSize, boxSize, CV_8UC1);
				// construct the points which will be mapped
				Point2f fromP[4];
				fromP[0] = Point2f(boxSize*j, boxSize*i); // top left
				fromP[1] = Point2f(boxSize*j, boxSize*(i+1) - 1); // top right
				fromP[2] = Point2f(boxSize*(j + 1) - 1, boxSize*(i + 1) - 1); // bottom right
				fromP[3] = Point2f(boxSize*(j + 1) - 1, boxSize*i); // bottom left
				
				warpPerspective(binarizedPuzzle, boxes[i][j], getPerspectiveTransform(fromP, toP), Size(boxSize, boxSize));
				boxes[i][j].at<uchar>(boxSize - 1, boxSize - 1) = 255;
				boxes[i][j] = extractLargestCC(boxes[i][j], true);
				
				// remove noise
				erode(boxes[i][j], boxes[i][j], kernel);
				dilate(boxes[i][j], boxes[i][j], kernel);
				resizeImg(boxes[i][j], boxes[i][j], 100, false);


				
				// center the digit
				Point2i min = getObjectMin(boxes[i][j]);
				Point2i max = getObjectMax(boxes[i][j]);

				int margin = 10;
				int length = boxes[i][j].rows;

				Point2f srcPi[4], dstPi[4];
				srcPi[0] = min;
				srcPi[1] = Point2i(max.x, min.y);
				srcPi[2] = max;
				srcPi[3] = Point2i(min.x, max.y);

				dstPi[0] = Point2i(margin, margin);
				dstPi[1] = Point2i(length - margin, margin);
				dstPi[2] = Point2i(length - margin, length - margin);
				dstPi[3] = Point2i(margin, length - margin);

				Mat centeredImg = Mat::zeros(Size(length, length), CV_8UC1);
				warpPerspective(boxes[i][j], centeredImg, getPerspectiveTransform(srcPi, dstPi), Size(length, length));
				
				erode(centeredImg, centeredImg, kernel);
				erode(centeredImg, centeredImg, kernel);
				thinning(centeredImg, centeredImg);
				boxes[i][j] = negative(centeredImg);

				//char name[MAX_PATH];
				//sprintf(name, "results/sudoku_image%d%d.png", i, j);
				//imwrite(name, boxes[i][j]);
				//imshow("box", boxes[i][j]);
				//waitKey();
			}
		}
		///////////////////////////////////////////
		// Step 6 - use template matching to recognize the digits from the elementar boxes
		
		// load the template images and compute the distance tranform
		char name[MAX_PATH];
		Mat templates[9];
		for (int k = 0; k < 9; k++){
			sprintf(name, "template/1_%d.png", k+1);
			templates[k] = imread(name, CV_LOAD_IMAGE_GRAYSCALE);

			distanceTransform(templates[k], templates[k], CV_DIST_L2, CV_DIST_MASK_5);
			normalize(templates[k], templates[k], 0, 1, NORM_MINMAX);

			//imshow("DT", templates[k]);
			//waitKey();
		}
		
		// superimpose the unknown image with the template and compute the pattern matching score
		for (int i = 0; i < 9; i++){
			for (int j = 0; j < 9; j++){

				double scores[9];
				for (int k = 0; k < 9; k++){
					scores[k] = patternMatchingScore(templates[k], boxes[i][j], 0);
					//printf("Template %d -- avg: %.4f\n", k + 1, scores[k]);
				}

				// find the minimul pattern matching score;
				if (scores[0] < 1.0){
					int min_index = 0;
					for (int k = 1; k < 9; k++){
						if (scores[k] < scores[min_index])
							min_index = k;
					}
					printf("%d ", min_index + 1);
				}
				else {
					printf("  ");
				}
				

				//printf("\n");
				//imshow("image", boxes[i][j]);
				//waitKey();
			}
			printf("\n");
		}
		
		///////////////////////////////////////////
		// Step 7 - solve the generated problem using backtracking

		///////////////////////////////////////////
		// Step 8 - draw the solution

		imshow("Original image", src);
		waitKey();

		// free the allocated memory for elementar boxes
		for (int i = 0; i < 9; i++)
			delete[]boxes[i];
		delete[]boxes;
	}
}

int main()
{
	startSolver();
	return 0;
}