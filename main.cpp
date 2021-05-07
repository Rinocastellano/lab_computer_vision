#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/video/tracking.hpp>
using namespace std;
using namespace cv;
vector<vector<Point2f>> matching(Mat frame,vector<Mat> objects){
    vector<cv::KeyPoint> keypoint;
    vector<vector<KeyPoint>> keypoints;
    Mat output;
    vector<Mat> detected_keypoints;
    Ptr<Feature2D> SiftPtr1=xfeatures2d::SIFT::create();
    vector<vector<Point2f>> scene_points_books;
    SiftPtr1->detect(frame,keypoint);
    SiftPtr1->compute(frame,keypoint,output);
    //imshow("output",output);
    //waitKey();
    Mat img1;
    drawKeypoints(frame,keypoint,img1);
    imshow("test features find",img1);
    waitKey();
    keypoints.push_back(keypoint);
    detected_keypoints.push_back(output);
    for(int i=0; i<objects.size();i++){

        Ptr<Feature2D> SiftPtr=xfeatures2d::SIFT::create();

        SiftPtr->detect(objects[i],keypoint);
        SiftPtr->compute(objects[i],keypoint,output);
        Mat img;
        drawKeypoints(objects[i],keypoint,img);
        keypoints.push_back(keypoint);
        detected_keypoints.push_back(output);
        //imshow("test features find", img);
        //waitKey();
    }

    Mat images;
for(int i=0;i<objects.size();i++){Ptr<BFMatcher> matcher= cv::BFMatcher::create(NORM_L2,false);
    vector<vector<DMatch>> matches;
    Mat d1=detected_keypoints[0].clone();
    Mat d2=detected_keypoints[i+1].clone();
    //cout<<d2<<endl;
    matcher->knnMatch(d2,d1,matches,2,noArray());

    Mat img_matches;
    //get good threshold
    const float ratio_thresh = 0.4;
    vector<DMatch> good_matches;
    for (size_t j = 0; j < matches.size(); j++)
    {
        if (matches[j][0].distance < ratio_thresh * matches[j][1].distance)
        {
            good_matches.push_back(matches[j][0]);
        }
    }

    drawMatches(objects[i],keypoints[i+1],frame,keypoints[0],
                good_matches,img_matches,Scalar::all(-1),
                Scalar::all(-1),std::vector<char>(),cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    //drawMatches(vecImg[i],keypoints[i],vecImg[i+1],keypoints[i+1],matches,img_matches);

    //imshow("matches",img_matches);
    //waitKey(0);
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    for( size_t j = 0; j < good_matches.size(); j++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( keypoints[i+1][ good_matches[j].queryIdx ].pt );
        scene.push_back( keypoints[0][ good_matches[j].trainIdx ].pt );
    }
    scene_points_books.push_back(scene);
    //
    Mat mask;
    Mat H = findHomography( obj, scene, RANSAC,5.0);



    //NON SICURO
    vector<Point2f> obj_corners(4);
    obj_corners[0] = Point2f(0, 0);
    obj_corners[1] = Point2f( (float)objects[0].cols-1, 0 );
    obj_corners[2] = Point2f( (float)objects[0].cols-1, (float)objects[0].rows-1 );
    obj_corners[3] = Point2f( 0, (float)objects[0].rows-1 );


    vector<Point2f> scene_corners(4);
    perspectiveTransform( obj_corners, scene_corners, H);
    //load the corners on the return var
    //corner_objects.push_back(scene_corners);

    cout<<scene_corners<<endl;

    for(int i=0;i<3;i++){
        line(frame,scene_corners[i],scene_corners[i+1],Scalar(0,0,255),5);
    }
    line(frame,scene_corners[3],scene_corners[0],Scalar(0,0,255),5);
    //imshow("found",frame);
    //waitKey();
}
return scene_points_books;
}

int main(){
    VideoCapture cap("./Lab 6 data/video.mov");
    if(cap.isOpened()){
        int i=0;
        Mat frame;
        vector<Mat> objects;
        vector<vector<Point2f>> scene_points;
        objects.push_back(imread("./Lab 6 data/objects/obj1.png"));
        objects.push_back(imread("./Lab 6 data/objects/obj2.png"));
        objects.push_back(imread("./Lab 6 data/objects/obj3.png"));
        objects.push_back(imread("./Lab 6 data/objects/obj4.png"));
        Mat prev_frame;
        for(;;){

            //OTHERWISE TRACKING
            if(!cap.read(frame))
                break;
            if(i==0){
                //if I'm here it means that I have to do matching
                scene_points=matching(frame,objects);
                //cout<<corners_objects[0]<<endl;
                prev_frame=frame.clone();
                i++;
            }
            else{
                vector<Point2f> new_scene_points;
                cout<<scene_points[0]<<endl;
                Mat frame_gray,prev_frame_gray;
                cvtColor(prev_frame, prev_frame_gray, COLOR_BGR2GRAY);
                cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
                vector<uchar> status;
                vector<float> err;
                TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
                cout<<scene_points[0]<<endl;
                calcOpticalFlowPyrLK(prev_frame_gray, frame_gray, scene_points[0], new_scene_points, status, err, Size(15,15), 2, criteria);
                cout<<new_scene_points<<endl;
                cout<<status[0]<<endl;
                vector<Point2f> good_new;
                Mat mask = Mat::zeros(frame.size(), frame.type());
                for(uint i = 0; i < scene_points.size(); i++)
                {
                    // Select good points
                    if(status[i] == 1) {
                        good_new.push_back(new_scene_points[i]);


                        circle(frame, new_scene_points[i], 5,Scalar(0,0,255), -1);
                    }
                    cout<<good_new<<endl;
                }
                imshow("result",frame);
                waitKey();
                /*_OutputArray features_found;
                vector<float> feature_errors;
                CvSize pyr_sz = cvSize( prev_frame.cols+8, frame.rows/3 );
                IplImage* pyrA = cvCreateImage( pyr_sz, IPL_DEPTH_32F, 1 );
                IplImage* pyrB = cvCreateImage( pyr_sz, IPL_DEPTH_32F, 1 );
                //vector<Point2f> corners;
                calcOpticalFlowPyrLK(prev_frame, frame,corners_objects[0],corners,features_found,feature_errors,cvSize( 10,10 )
                                       ,3,cvTermCriteria( CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, .3 ),0);
                cout<<corners<<endl;
                imshow("result",frame);
                waitKey();*/
            }
            imshow("window",frame);
            char key= cvWaitKey();
            if(key==27)
                break;
        }
    }
    return 0;
}
