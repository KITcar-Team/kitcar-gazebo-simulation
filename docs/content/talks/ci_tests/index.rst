.. _talk_ci_tests:

Continuous Integration using the Simulation
===========================================

.. tip::

   A recording of this talk can be found under the following link:
   https://webdav.kitcar-team.de/Workshops/2020-07-Workshop-Samstag/Workshop_Simulation_in_der_CI/Workshop_simulation_ci.mp4

.. raw:: html

   <video width="100%" class="video-background" autoplay loop muted playsinline>
     <source src="../../drive_test_obstacles.mp4" type="video/mp4">
   Your browser does not support the video tag.
   </video>

|

.. raw:: html

   <video width="100%" class="video-background" autoplay loop muted playsinline>
     <source src="../../drive_test_failure.mp4" type="video/mp4">
   Your browser does not support the video tag.
   </video>

|

What are the goals of this workshop?

* Take you behind the scenes of the video above,
* show you how the video and our continuous integration connect,
* discuss **together** how we should continue.

.. toctree::
   :maxdepth: 1
   :caption: Content

   .. road Modular + Python Script + Power of Python
   .. groundtruth Extrahiert aus der road + Verfuegbar ueber ROS services
   .. state_machines Zustand der Fahrt tracken und bewerten
   .. drive_test Verpackt in ROS test
   .. ci Integriert in die CI pipeline
   road
   groundtruth
   evaluation
   drive_test
   ci

.. _talk_ci_tests_outlook:

Outlook: Possibilities
----------------------

To us, it feels like we can finally leverage all of the power the simulation brings.
While there are many ways we can go from here, not all are equally obvious and valuable.

A blatantly obvious and small step is to

* *Write* **more** *tests on* **more** *roads to cover* **more**...

However, there are more ambitious and long term plans, we could pursue:

* Create a **test suite** and produce a detailed report of mistakes that were made
* Write a search routine to explicitly search for mistakes

Things to consider:

* *Proceed carefully to ensure that the test results are representative of the car's real behavior.*

Questions + Discussion
----------------------
