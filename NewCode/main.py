from flask import Flask, Response, render_template
from dynamic_traffic_control import process_video

app = Flask(__name__)

@app.route('/')
def index():
    """
    Home route to render the HTML page that will display two video streams.
    """
    return render_template('index.html')


@app.route('/video_stream')
def video_stream():
    """
    Video stream for the first side of the road.
    """
    return Response(process_video('TrafficVideo.mp4', current_signal="green"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_stream1')
def video_stream1():
    """
    Video stream for the second side of the road.
    """
    return Response(process_video('TrafficVideo1.mp4', current_signal="red"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    # Run the Flask application
    app.run(host='0.0.0.0', port=5000, debug=False)
