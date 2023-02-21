import numpy as np
import datetime
from time import sleep
import cv2
import os
import json
import pathlib
import numpy as np
from queue import Queue
from threading import Thread
from flask import Flask, Response, render_template
from kafka import KafkaConsumer
from deep_sort.deep_sort.tracker import Tracker
from pathlib import Path
import os
import random
import time
from calc_speed import calcSpeed
from deep_sort.deep_sort import preprocessing, nn_matching
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort.tracker import Tracker
import matplotlib.pyplot as plt
from flask import Flask, render_template, make_response, request, jsonify

# import from helpers
from tracking_helpers import read_class_names, create_box_encoder
from detection_helpers import *

class DetectionTrackingModel():
    def __init__(self, model_path,
                reID_model_path,
                max_cosine_distance=0.4, 
                nms_max_overlap=1.0, 
                coco_names_path="./io_data/input/classes/coco.names"):

        # YOLO v7
        self.detector = Detector(conf_thres=0.25)
        self.detector.load_model(model_path, trace=False)

        self.coco_names_path = coco_names_path
        self.nms_max_overlap = nms_max_overlap
        self.class_names = read_class_names()

        # initialize Deep Sort
        self.encoder = create_box_encoder(reID_model_path, batch_size=128)
        # device = select_device("0" if torch.cuda.is_available() else 'cpu')
        # self.encoder = torch.load(reID_model_path, map_location=torch.device(device))
        # self.encoder = self.encoder.eval()
        
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.4, None)
        self.tracker = Tracker(metric)
        self.count_objects = True
        self.verbose = 1

        self.save_to = './detected_frame'
        pathlib.Path(self.save_to).mkdir(parents=True, exist_ok=True)

    def save_txt(self, frame_num, bboxes, scores, classes):
        file_path = f'{self.save_to}/{frame_num}.txt'
        with open(file_path, 'w') as f:
            for bbox, score, class_id in zip(bboxes, scores, classes):
                f.write(f"{int(bbox[0])},{int(bbox[1])},{int(bbox[2])},{int(bbox[3])},{score},{int(class_id)}")
                f.write('\n')

    def load_txt(self, frame_num):
        bboxes = []
        scores = []
        classes = []
        print("Loading detected frame from file...")
        file_path = f'{self.save_to}/{frame_num}.txt'
        with open(file_path) as f:
            for line in f:
                detect_data = line.split(',')
                bboxes.append([int(detect_data[0]), int(detect_data[1]), int(detect_data[2]), int(detect_data[3])])
                scores.append(float(detect_data[4]))
                classes.append(int(detect_data[5]))
        num_objects = len(bboxes)

        return bboxes, scores, classes, num_objects

    def detect_and_tracking(self, frame, frame_num, skip_frames=0):
        # skip every nth frame. When every frame is not important, you can use this to fasten the process
        if skip_frames and not frame_num % skip_frames:
            return False, frame
        
        if self.verbose >= 1:
            start_time = time.time()

        if not os.path.exists(f'./detected_frame/{frame_num}.txt'):
            # Get the detections
            yolo_dets = self.detector.detect(frame.copy(), plot_bb = False)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if yolo_dets is None:
                bboxes = []
                scores = []
                classes = []
                num_objects = 0
            
            else:
                bboxes = yolo_dets[:,:4]
                bboxes[:,2] = bboxes[:,2] - bboxes[:,0] # convert from xyxy to xywh
                bboxes[:,3] = bboxes[:,3] - bboxes[:,1]

                scores = yolo_dets[:,4]
                classes = yolo_dets[:,-1]
                num_objects = bboxes.shape[0]
            
            self.save_txt(frame_num, bboxes, scores, classes)
        
        else:
            bboxes, scores, classes, num_objects = self.load_txt(frame_num)

        # ---------------------------------------- DETECTION PART COMPLETED ---------------------------------------------------------------------
        
        names = []
        for i in range(num_objects): # loop through objects and use class index to get class name
            class_indx = int(classes[i])
            class_name = self.class_names[class_indx]
            names.append(class_name)

        names = np.array(names)
        count = len(names)

        if self.count_objects:
            cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, ((255, 255, 255)), 2)

        # ---------------------------------- DeepSORT tacker work starts here ------------------------------------------------------------
        features = self.encoder(frame, bboxes) # encode detections and feed to tracker. [No of BB / detections per frame, embed_size]
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)] # [No of BB per frame] deep_sort.detection.Detection object

        cmap = plt.get_cmap('tab20b') # initialize color map
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        boxs = np.array([d.tlwh for d in detections])  # run non-maxima supression below
        scores = np.array([d.confidence for d in detections])
        classes = np.array([d.class_name for d in detections])
        indices = preprocessing.non_max_suppression(boxs, classes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]       

        self.tracker.predict()  # Call the tracker
        self.tracker.update(detections) #  updtate using Kalman Gain

        for track in self.tracker.tracks:  # update new findings AKA tracks
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()

            # Calculate speed of object
            track = calcSpeed(track, bbox, frame_num, 30)
            
            # draw bbox on screen
            color = colors[int(track.track_id) % len(colors)]  
            color = [i * 255 for i in color]
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            text_header_bbox: str = class_name + ":" + str(track.track_id)

            if track.speed > 0:
                print(f"{track.class_name} {track.track_id}: {track.speed} km/h")
                text_header_bbox += "-" + str(round(track.speed, 1)) + "km/h"

            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1] - 30)),
                        (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17, int(bbox[1])), color,
                        -1)
            cv2.putText(frame, text_header_bbox, (int(bbox[0]), int(bbox[1] - 11)), 0, 0.6,
                        (255, 255, 255), 1, lineType=cv2.LINE_AA)    

            if self.verbose == 2:
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
                
        # -------------------------------- Tracker work ENDS here -----------------------------------------------------------------------
        if self.verbose >= 1:
            fps = 1.0 / (time.time() - start_time) # calculate frames per second of running detections
            if not self.count_objects: print(f"Processed frame no: {frame_num} || Current FPS: {round(fps, 2)}")
            else: print(f"Processed frame no: {frame_num} || Current FPS: {round(fps,2)} || Objects tracked: {count}")
        
        frame_result = np.asarray(frame)
        frame_result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # cv2.imshow('frame', result)
        return True, frame_result
        
class KafkaVideoView():
    def __init__(self, bootstrap_servers, topic, client_id, group_id, poll=500, frq=0.01):
        self.topic = topic
        self.client_id = client_id
        self.group_id = group_id
        self.bootstrap_servers = bootstrap_servers
        self.poll = poll
        self.frq = frq
        self.frame_num = 0

        self.detector_and_tracker = DetectionTrackingModel(model_path='./weight/best.pt',
                                                           reID_model_path='./deep_sort/model_weights/mars-small128.pb')

    def setConsumer(self):
        self.consumer = KafkaConsumer(
                self.topic, 
                bootstrap_servers=self.bootstrap_servers.split(','),
                fetch_max_bytes=52428800,
                fetch_max_wait_ms=1000,
                fetch_min_bytes=1,
                max_partition_fetch_bytes=1048576,
                value_deserializer=None,
                key_deserializer=None,
                max_in_flight_requests_per_connection=10,
                client_id=self.client_id,
                group_id=self.group_id,
                auto_offset_reset='earliest',
                max_poll_records=self.poll,
                max_poll_interval_ms=300000,
                heartbeat_interval_ms=3000,
                session_timeout_ms=10000,
                enable_auto_commit=True,
                auto_commit_interval_ms=5000,
                reconnect_backoff_ms=50,
                reconnect_backoff_max_ms=500,
                request_timeout_ms=305000,
                receive_buffer_bytes=32768,
            )

    def playStream(self, queue):
        while self.keepPlaying:
            try:
                msg = queue.get(block=True, timeout=20)
                self.queue_status = True
            except:
                print("WARN: Timed out waiting for queue. Retrying...")
                self.queue_status = False

            if self.queue_status:
                self.frame_num += 1
                print(f"Processing frame {self.frame_num}...")
                nparr = np.frombuffer(msg, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                isSuccess, result_frame = self.detector_and_tracker.detect_and_tracking(frame, self.frame_num)
                cv2.imshow('Streaming Video', result_frame if isSuccess else frame)
            
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.keepConsuming = False
                    cv2.destroyAllWindows()
                    break

                sleep(self.frq)

    def run(self):
        self.keepPlaying = True
        self.setConsumer()
        self.videoQueue = Queue()
        self.keepConsuming = True

        self.playerThread = Thread(target=self.playStream, args=(self.videoQueue, ), daemon=False)
        self.playerThread.start()

        try:
            Path("./detected_frame").mkdir(parents=True, exist_ok=True)
            while self.keepConsuming:
                payload = self.consumer.poll(self.poll)
                for bucket in payload:
                    for msg in payload[bucket]:
                        self.videoQueue.put(msg.value)

        except KeyboardInterrupt:
            self.keepConsuming = False
            self.keepPlaying = False
            cv2.destroyAllWindows()
            print("WARN: Keyboard Interrupt detected. Exiting...")

        self.playerThread.join()


topic = "KafkaVideoStream"
consumer = KafkaConsumer(
    topic, 
    bootstrap_servers=['localhost:9092'])

app = Flask(__name__)

count_car = 0
count_van = 0
count_bus = 0
count_truck = 0

labels_line = ['0']
values_line_car = [0]
values_line_van = [1]
values_line_bus = [2]
values_line_truck = [5]

@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        conf_thres = request.data
        detector_and_tracker.detector.conf_thres = int(conf_thres)/100
        print("Confidence threshold", conf_thres)
    return render_template('index1.html',
                           count_car=count_car,
                           count_van=count_van,
                           count_bus=count_bus,
                           count_truck=count_truck,
                           labels_line=labels_line,
                            values_line_car=values_line_car,
                            values_line_van=values_line_van, 
                            values_line_bus=values_line_bus, 
                            values_line_truck=values_line_truck)

@app.route('/refreshData')
def refresh_graph_data():
    global count_car, count_van, count_bus, count_truck
    global labels_line, values_line_car, values_line_van, values_line_bus, values_line_truck
    return jsonify(count_car=count_car,
                   count_van=count_van,
                   count_bus=count_bus,
                   count_truck=count_truck,
                   labels_line=labels_line[-5:],
                   values_line_car=values_line_car[-5:],
                   values_line_van=values_line_van[-5:], 
                   values_line_bus=values_line_bus[-5:], 
                   values_line_truck=values_line_truck[-5:])

@app.route('/data', methods=["GET", "POST"])
def data():
    data = [time.time() * 1000, random.random() * 100]
    data1 = [time.time() * 1000, random.random() * 100]
    data2 = [time.time() * 1000, random.random() * 100]
    data3 = [time.time() * 1000, random.random() * 100]
    response = make_response(json.dumps([data, data1, data2, data3]))
    response.content_type = 'application/json'
    return response

@app.route('/video_feed', methods=['GET'])
def video_feed():
    return Response(
        get_video_stream(), 
        mimetype='multipart/x-mixed-replace; boundary=frame')

detector_and_tracker = DetectionTrackingModel(model_path='./weight/best.pt',
                                              reID_model_path='./deep_sort/deep_sort/model_weights/mars-small128.pb')
frame_num = 0

def get_video_stream():
    global frame_num
    global count_car, count_van, count_bus, count_truck
    global labels_line, values_line_car, values_line_van, values_line_bus, values_line_truck
    for message in consumer:
        nparr = np.frombuffer(message.value, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        _, result_frame = detector_and_tracker.detect_and_tracking(frame, frame_num)
        count_car += 1
        count_van += 1
        count_bus += 1
        count_truck += 1
        frame_num += 1

        if frame_num % 30 == 0:
            labels_line.append('0')
            values_line_car.append(np.random.randint(1,10))
            values_line_van.append(np.random.randint(1,10))
            values_line_bus.append(np.random.randint(1,10))
            values_line_truck.append(np.random.randint(1,10))

        ret, buffer = cv2.imencode('.jpg', result_frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + buffer.tobytes() + b'\r\n\r\n')


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)

    # streamVideoPlayer = KafkaVideoView(
    #     bootstrap_servers='localhost:9092',
    #     topic='KafkaVideoStream',
    #     client_id='KafkaVSClient',
    #     group_id='KafkaVideoStreamConsumer',
    #     poll=500,
    #     frq=0.025
    # )

    # streamVideoPlayer.run()
