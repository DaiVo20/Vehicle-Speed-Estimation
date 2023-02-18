import datetime
from time import sleep
import cv2
import time
import numpy as np
from queue import Queue
from threading import Thread
from flask import Flask, Response, render_template
from kafka import KafkaConsumer
from deep_sort.tracker import Tracker
from pathlib import Path
import os
from calc_speed import calculate_speed
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
import matplotlib.pyplot as plt

# import from helpers
from tracking_helpers import read_class_names, create_box_encoder
from detection_helpers import *

topic = "distributed-video"
consumer = KafkaConsumer(
    topic, 
    bootstrap_servers=['localhost:9092'])

app = Flask(__name__)

class KafkaVideoView():
    def __init__(self, bootstrap_servers, topic, client_id, group_id, poll=500, frq=0.01):
        self.topic = topic
        self.client_id = client_id
        self.group_id = group_id
        self.bootstrap_servers = bootstrap_servers
        self.poll = poll
        self.frq = frq
        self.frame_num = 0
        
        detector = Detector(classes=[2, 5, 6, 7])
        detector.load_model('yolov7x.pt')

        # YOLOv7 DeepSort
        self.detector = detector
        self.coco_names_path = "./io_data/input/classes/coco.names"
        self.nms_max_overlap = 1.0
        self.class_names = read_class_names()

        # initialize deep sort
        self.encoder = create_box_encoder("./deep_sort/model_weights/mars-small128.pb", batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.4, None) # calculate cosine distance metric
        self.tracker = Tracker(metric) # initialize tracker

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
                nparr = np.frombuffer(msg, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                verbose = 1
                count_objects = True

                if verbose >= 1:
                    start_time = time.time()
                
                print(self.frame_num)

                if not os.path.exists(f'./detected_frame/{self.frame_num}.txt'):
                    yolo_dets = self.detector.detect(frame.copy(), plot_bb = False)  # Get the detections
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

                        with open(f'./detected_frame/{self.frame_num}.txt', 'w') as f:
                            for bbox, score, class_id in zip(bboxes, scores, classes):
                                f.write(f"{int(bbox[0])},{int(bbox[1])},{int(bbox[2])},{int(bbox[3])},{score},{int(class_id)}")
                                f.write('\n')
                else:
                    bboxes = []
                    scores = []
                    classes = []
                    print("Loading detected frame from file...")
                    with open(f'./detected_frame/{self.frame_num}.txt') as f:
                        for line in f:
                            detect_data = line.split(',')
                            bboxes.append([int(detect_data[0]), int(detect_data[1]), int(detect_data[2]), int(detect_data[3])])
                            scores.append(float(detect_data[4]))
                            classes.append(int(detect_data[5]))
                    num_objects = len(bboxes)

                # ---------------------------------------- DETECTION PART COMPLETED ---------------------------------------------------------------------
        
                names = []
                for i in range(num_objects): # loop through objects and use class index to get class name
                    class_indx = int(classes[i])
                    class_name = self.class_names[class_indx]
                    names.append(class_name)

                names = np.array(names)
                count = len(names)

                if count_objects:
                    cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, ((255, 255, 255)), 2)

                # ---------------------------------- DeepSORT tacker work starts here ------------------------------------------------------------
                features = self.encoder(frame, bboxes) # encode detections and feed to tracker. [No of BB / detections per frame, embed_size]
                detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)] # [No of BB per frame] deep_sort.detection.Detection object

                cmap = plt.get_cmap('tab20b') #initialize color map
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

                    track = calculate_speed(track, [(646, 341), (1111, 327)], [(601, 412), (1166, 392)], bbox, self.frame_num)
                    
                    color = colors[int(track.track_id) % len(colors)]  # draw bbox on screen
                    color = [i * 255 for i in color]
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
                    
                    if track.speed > 0:
                        print(f"{track.class_name} {track.track_id}: {track.speed} km/h")
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0]) + (len(class_name) + len(str("-" + str(round(track.speed, 2)) + "km/h")) + len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                        cv2.putText(frame, class_name + " : " + str(track.track_id) + " - " + str(round(track.speed, 2)) + "km/h",(int(bbox[0]), int(bbox[1]-11)),0, 0.6, (255,255,255),1, lineType=cv2.LINE_AA) 
                    else:
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(bbox[1])), color, -1)
                        cv2.putText(frame, class_name + " : " + str(track.track_id),(int(bbox[0]), int(bbox[1]-11)),0, 0.6, (255,255,255),1, lineType=cv2.LINE_AA)    

                    if verbose == 2:
                        print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
                        
                # -------------------------------- Tracker work ENDS here -----------------------------------------------------------------------
                if verbose >= 1:
                    fps = 1.0 / (time.time() - start_time) # calculate frames per second of running detections
                    if not count_objects: print(f"Processed frame no: {self.frame_num} || Current FPS: {round(fps, 2)}")
                    else: print(f"Processed frame no: {self.frame_num} || Current FPS: {round(fps,2)} || Objects tracked: {count}")
                
                result = np.asarray(frame)
                result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                cv2.imshow('frame', result)
            
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


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed', methods=['GET'])
def video_feed():
    global streamVideoPlayer
    return Response(
        get_video_stream(), 
        mimetype='multipart/x-mixed-replace; boundary=frame')

def get_video_stream():
    for msg in consumer:
        yield (b'--frame\r\n'
               b'Content-Type: image/jpg\r\n\r\n' + msg.value + b'\r\n\r\n')


if __name__ == "__main__":
    # app.run(host='0.0.0.0', debug=True)
    streamVideoPlayer = KafkaVideoView(
        bootstrap_servers='localhost:9092',
        topic='KafkaVideoStream',
        client_id='KafkaVSClient',
        group_id='KafkaVideoStreamConsumer',
        poll=500,
        frq=0.025
    )

    streamVideoPlayer.run()