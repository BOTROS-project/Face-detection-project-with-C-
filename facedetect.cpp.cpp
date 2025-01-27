#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main() {
    // Open the default camera (usually the built-in webcam)
    VideoCapture video(0);
    if (!video.isOpened()) {
        cerr << "Oops! Couldn't open the camera." << endl;
        return -1;
    }

    // Load the face detection model
    CascadeClassifier faceDetector;
    if (!faceDetector.load("haarcascade_frontalface_default.xml")) {
        cerr << "Uh-oh! Couldn't load the Haar Cascade model." << endl;
        return -1;
    }

    Mat frame;
    while (true) {
        // Capture the current frame from the camera
        video.read(frame);
        if (frame.empty()) {
            cerr << "Oops! Captured an empty frame." << endl;
            break;
        }

        // Detect faces in the frame
        vector<Rect> faces;
        faceDetector.detectMultiScale(frame, faces, 1.3, 5);

        // Draw rectangles around each detected face
        for (const auto& face : faces) {
            rectangle(frame, face, Scalar(50, 50, 255), 3);
            rectangle(frame, Point(0, 0), Point(250, 70), Scalar(50, 50, 255), FILLED);
            putText(frame, to_string(faces.size()) + " Face(s) Found", Point(10, 40), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 255, 255), 2);
        }

        // Display the frame with detected faces
        imshow("Face Detection", frame);

        // Wait for 1 ms and break the loop if 'q' is pressed
        if (waitKey(1) == 'q') {
            break;
        }
    }

    return 0;
}
