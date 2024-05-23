# YOLOv8 Batching and Segmentation with ONNX.js

This project performs batching for YOLOv8 models using the ONNX.js format, enabling simultaneous image segmentation on multiple images. It efficiently processes any number of images in a single batch.

## Features
- **Batch Processing:** Supports batching of multiple images for simultaneous processing.
- **YOLOv8 Model:** Utilizes the YOLOv8 model for object detection and segmentation.
- **ONNX.js Integration:** Leverages ONNX.js for running the model in a web environment.
- **Flexible Input:** Accepts any number of images for processing in a single batch.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/Aisha-Hassan1/yolov8-batching-onnxjs.git
    ```
2. Navigate to the project directory:
    ```bash
    cd yolov8-batching-onnxjs
    ```
3. Install the required dependencies:
  
    ```

## Usage
1. Place the YOLOv8 ONNX model file in the `models` directory.
2. Prepare your images and place them in the `images` directory.
3. Run the application:
   
4. Access the application in your web browser at `http://localhost:3000`.
5. Upload your images through the web interface and start the batching and segmentation process.

## Project Structure
- `model/`: Directory containing the YOLOv8 ONNX model.
- `img/`: Directory for input images to be processed.
- `index.html`: Main HTML file for the web interface.

## Example
Here's an example of how to use the application:
1. Upload multiple images through the web interface.
2. Click the "Process Batch" button.
3. View the segmented images with detected objects highlighted.

## Dependencies
- **ONNX:** Library for running ONNX models in the browser.
- **YOLOv8:** Pre-trained YOLOv8 model in ONNX format.

## Resources
- [ONNX.js Documentation](https://github.com/microsoft/onnxjs)
- [YOLOv8 Model](https://github.com/ultralytics/yolov8)

## License
This project is licensed under the MIT License.

---

This project aims to provide a robust solution for batching and segmenting images using YOLOv8 and ONNX.js. If you have any questions or feedback, feel free to reach out!
