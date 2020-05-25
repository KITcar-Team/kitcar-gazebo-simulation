import os

import yaml
import rospy
import rospkg

from python_qt_binding import loadUi

from qt_gui.plugin import Plugin
from python_qt_binding.QtWidgets import QPushButton, QWidget


from std_msgs.msg import Empty as EmptyMsg
from simulation_groundtruth.msg import GroundtruthStatus


class SimulationRendererPlugin(Plugin):
    """Basic rqt plugin that allows to start/stop a reload in the renderer."""

    def __init__(self, context):
        super(SimulationRendererPlugin, self).__init__(context)

        self._widget = QWidget()

        ui_file = os.path.join(
            rospkg.RosPack().get_path("simulation_rqt"), "resource", "SimulationRenderer.ui"
        )
        loadUi(ui_file, self._widget)
        context.add_widget(self._widget)

        # Buttons
        self.button_stop = self._widget.findChild(QPushButton, "buttonStop")
        self.button_reload = self._widget.findChild(QPushButton, "buttonReload")

        # GUI Callbacks
        self.button_reload.clicked.connect(self.reload_renderer)
        self.button_stop.clicked.connect(self.stop_renderer)

        groundtruth_topics = self._load_groundtruth_topics()
        renderer_topics = groundtruth_topics["renderer"]
        self.reload_publisher = rospy.Publisher(
            renderer_topics["reload"], EmptyMsg, queue_size=1
        )
        self.stop_publisher = rospy.Publisher(
            renderer_topics["interrupt"], EmptyMsg, queue_size=1
        )

        self.info_subscriber = rospy.Subscriber(
            groundtruth_topics["status"],
            GroundtruthStatus,
            queue_size=1,
            callback=self.receive_groundtruth_status,
        )

        self.button_reload.setEnabled(True)
        self.button_stop.setEnabled(False)

    def _load_groundtruth_topics(self):
        topic_file = os.path.join(
            rospkg.RosPack().get_path("simulation_groundtruth"),
            "param",
            "groundtruth",
            "topics.yaml",
        )
        with open(topic_file, "r") as file:
            return yaml.safe_load(file)

    def receive_groundtruth_status(self, msg):
        """Receive new status update from renderer."""
        self.button_reload.setEnabled(msg.status == GroundtruthStatus.READY)
        self.button_stop.setEnabled(
            msg.status == GroundtruthStatus.REMOVE_OLD_TILES
            or msg.status == GroundtruthStatus.RENDER_NEW_TILES
        )

    def reload_renderer(self):
        """Publish message on renderer reload topic."""
        self.reload_publisher.publish()

    def stop_renderer(self):
        """Publish message on renderer stop topic."""
        self.stop_publisher.publish()
