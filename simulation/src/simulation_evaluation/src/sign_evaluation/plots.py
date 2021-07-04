import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import rospy


def create_plot(
    kinds,
    values,
    save_dir,
    y_label="Score",
    file_name="eval",
    title="Evaluation of the Test",
):
    """Save the plots to the file system.

    Args:
        kinds: The class kinds of the bars.
        values: Height of each bar.
        save_dir: Save directory for the plots
    """
    if len(values) == 0:
        rospy.logwarn("Zero detections found")
        return

    x_markings = kinds + ["mean"]
    xs = list(range(0, len(x_markings)))
    width = 0.35
    fig, ax = plt.subplots()
    rospy.loginfo(values)
    ax.barh(xs[:-1], values, width)
    ax.barh(xs[-1], np.mean(values), width)
    ax.set_yticks(xs)
    ax.set_yticklabels(x_markings)
    ax.set_xlabel(y_label)
    ax.set_title(title)
    fig.set_size_inches(20, max(5, 0.4 * len(x_markings)))
    now = datetime.now().replace(second=0, microsecond=0)
    time_string = f"_{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}"
    plt.subplots_adjust(left=0.2)
    rospy.logdebug(f"Saving plot {file_name}")
    plt.savefig(os.path.join(save_dir, file_name + time_string + ".pdf"))
