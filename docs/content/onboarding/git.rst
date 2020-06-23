.. _onboarding_git:

Git
===

Before jumping into the details of **kitcar-gazebo-simulation** \
we need to take a second to talk about workflow.
Writing and maintaining a large software project with multiple collaborators is challenging.
Luckily, we use several great tools at KITcar.

The Onboarding contains multiple tasks that you need to solve.
You will use Git to create your branch to work on \
and commit your solutions to share them with others.

Create a Branch
---------------

After cloning the repository, you are on the **master** branch by default.
The **master** is protected and may only be modified through merge requests.
You, therefore, need to create a new branch \
whenever you want to make changes within the repository.

You should now open a terminal and change into the kitcar-gazebo-simulation folder.
Here, your first task begins.
Within the general Onboarding, you have already learned how to create and \
checkout git branches:

.. admonition:: Your Task

   Create and checkout a new branch with the name **onboarding_<YOURNAME>**.

Commit
------

Throughout the Onboarding, you will be asked to commit your changes. You have already learned how to use

.. code-block:: shell

   git add

and

.. code-block:: shell

   git commit

and therefore know all the basics. Please refer to `How to Write a Git Commit Message <https://chris.beams.io/posts/git-commit/>`_ on how to write nice commit messages.

.. note::

   When installing **kitcar-gazebo-simulation** `pre-commit <https://pre-commit.com/>`_ was installed as well.
   Whenever you commit Python code, *pre-commit* will check if your code complies with our style guidelines.
   You will not be able to commit as long as you do not adhere to the guidelines.
   This means that you might have to modify your code.

   **Why?** Maintaining a clear and consistent style throughout a large software project is not easy.
   Pre-commit helps us to do so! (It can be annoying, we know...)



Push
----

Of course, everything up to this point happened locally on your computer.
Don't forget to use to

.. code-block:: shell

   git push

to upload your changes after commiting
