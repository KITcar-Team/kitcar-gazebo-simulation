Evaluation
==========

The evaluation is done using multiple stages.

Speaker: Car's Position + Groundtruth
-------------------------------------

Create a simple interpretation of what's happening.

.. raw:: html

   <video width="100%" class="video-background" autoplay loop muted playsinline>
     <source src="../../speaker_output.mp4" type="video/mp4">
   Your browser does not support the video tag.
   </video>

|


State Machines
--------------

At the evaluation's core are multiple *single-purpose* state machines that keep track of what's happening:

  * **Progress**: Whether the car is at the beginning/middle/end of the road
  * **Overtaking**: Whether the car correctly overtakes obstacles
  * ...

Example: OvertakingStateMachine
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   .. raw:: html

      <video width="100%" class="video-background" autoplay loop muted playsinline>
        <source src="../../state_machine_output.mp4" type="video/mp4">
      Your browser does not support the video tag.
      </video>

   |

   .. figure:: ../../simulation_evaluation/graphs/overtaking.svg
      :scale: 10 %
      :align: center
      :alt: Graph of OvertakingStateMachine

      Graph of OvertakingStateMachine

Referee
-------

The output of the state machines is monitored by a referee node that
check's if the

* state_machines are in valid states -> *Referee.DRIVING*
* car reaches the end of the road -> *Referee.COMPLETED*
* car makes mistake -> *Referee.FAILED*



Example: Referee Output
^^^^^^^^^^^^^^^^^^^^^^^

.. raw:: html

   <video width="100%" class="video-background" autoplay loop muted playsinline>
     <source src="../../drive_test_referee.mp4" type="video/mp4">
   Your browser does not support the video tag.
   </video>

|

The Complete Picture
--------------------

.. include:: ../../simulation_evaluation/index.rst
   :start-after: evaluation_pipeline_graph_start
   :end-before: evaluation_pipeline_graph_end


See :ref:`simulation_evaluation` for more details.
