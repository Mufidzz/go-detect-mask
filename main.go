package main

import (
	"fmt"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"gocv.io/x/gocv"
	"image"
	"image/color"
	"log"
	"path"
)

func main() {
	currPath := "/media/fz/Project (Running)/Unknown/TryDetectMask/face_detector"
	prototxtPath := path.Join(currPath, "deploy.prototxt")
	weightsPath := path.Join(currPath, "res10_300x300_ssd_iter_140000.caffemodel")
	definedConfidence := float32(0.5)

	modelPath := "mask_detector.model"

	webcam, _ := gocv.VideoCaptureDevice(0)
	window := gocv.NewWindow("Hello")

	net := gocv.ReadNet(prototxtPath, weightsPath)

	_, err := tf.LoadSavedModel(modelPath, []string{"serve"}, nil)

	if err != nil {
		log.Fatal("ERROR LOADING KERAS MODEL")
	}

	img := gocv.NewMat()
	imgOri := gocv.NewMat()

	for {
		webcam.Read(&img)
		img.CopyTo(&imgOri)

		imgSize := img.Size()

		img.ConvertTo(&img, gocv.MatTypeCV32F)
		imgBlob := gocv.BlobFromImage(img, 1.0, image.Point{X: 300, Y: 300}, gocv.NewScalar(104.0, 177.0, 123.0, 0), false, false)

		net.SetInput(imgBlob, "")
		detections := net.Forward("")

		for i := 0; i < detections.Total(); i += 7 {
			confidence := detections.GetFloatAt(0, i+2)

			if confidence > definedConfidence {

				fmt.Println(confidence)

				left := int(detections.GetFloatAt(0, i+3) * float32(imgSize[1]))
				top := int(detections.GetFloatAt(0, i+4) * float32(imgSize[0]))
				right := int(detections.GetFloatAt(0, i+5) * float32(imgSize[1]))
				bottom := int(detections.GetFloatAt(0, i+6) * float32(imgSize[0]))

				fmt.Println(left, top, right, bottom)

				gocv.Rectangle(&imgOri, image.Rect(left, top, right, bottom), color.RGBA{0, 0, 255, 0}, 4)

				//model.Graph.

			}
		}

		detections.Close()
		imgBlob.Close()

		window.IMShow(imgOri)
		window.WaitKey(1)
	}

}
