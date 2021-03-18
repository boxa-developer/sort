import cv2
import numpy as np
import timeit


# import myTracker


class Detector:
    def __init__(self):
        super().__init__()
        self.net = None
        self.output_layers = None
        self.setModels()
        self.classes = []
        with open("./yolo_weights/coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

    def setModels(self):
        print('SetModel:::::::::')
        net = cv2.dnn.readNet("./yolo_weights/yolov3-full.weights", "./yolo_weights/yolov3-full.cfg")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        self.net = net
        self.output_layers = output_layers

    def drawBoxes(self, frame):
        st = cv2.getTickCount()
        cv2.putText(frame, 'Detected Cars:  ', (250, 1210), cv2.FONT_HERSHEY_SIMPLEX, 2.75,
                    (0, 0, 255), 4)
        height, width, channels = frame.shape
        net, output_layers = self.net, self.output_layers
        # 0.00392
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (256, 256), (0, 0, 0), True, crop=False)
        net.setInput(blob)

        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        # print([box.append(indexes[i]) for i, box in enumerate(boxes)])
        colors = np.random.uniform(0, 255, size=(len(boxes), 3))
        dx = 30
        # print(self.classes)
        # temp = self.cntObj.update(boxes)
        dets = []
        for i in range(len(boxes)):
            if i in indexes:
                label = str(self.classes[class_ids[i]])
                if label == 'car':
                    dets.append(boxes[i] + [i])
                    # print(boxes[i]+[i])
                    x, y, w, h = boxes[i]

                    color = colors[i]

                    sX = x + w // 2 - dx
                    sY = y + h // 2 - dx
                    eX = x + w // 2 + dx
                    eY = y + h // 2 + dx

                    # cv2.rectangle(frame, (sX, sY), (eX, eY), color, 1)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
                    cv2.putText(frame, str(label), (sX, sY), cv2.FONT_HERSHEY_SIMPLEX, 1.75,
                                (0, 255, 0), 1)
                    cv2.putText(frame, 'Detected Cars: {}'.format(str(len(indexes))), (250, 1210),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2)
                    # # print()
                    # fps = cv2.getTickFrequency() / (cv2.getTickCount() - st)
                    # cv2.putText(frame, 'FPS: {}'.format(str(round(fps, 2))), (250, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                    #             2)
                    # cv2.imshow("Frame", frame)
                    # cv2.waitKey(1)
        return frame
