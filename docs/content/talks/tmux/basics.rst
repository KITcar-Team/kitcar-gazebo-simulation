The Basics
==========

What is Tmux?
-------------

Tmux is a terminal multiplexer. It means that you can access multiple terminal sessions
in a single window! In short: You only need to open one terminal instead of many.

It can easily become your IDE on the command line while running on every unix system
wheather local or a server. Tmux is also scriptable!


How to Install
--------------

Follow the instructions `here <https://github.com/tmux/tmux/wiki>`_

How to Use
----------

For starters: Tmux can create multiple sessions. Each session can have multiple windows
and each window can have multiple panes. For more details watch my recording.

.. note::

    You can read the manual page at any time with

    .. code-block::

        man tmux


The Most Important Commands
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: none

    tmux            // Start a new session
    tmux a          // Attatch to most recent session
    tmux ls         // List all active sessions

The Most Important Shortcuts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In tmux you have a prefix key (Default: Ctrl-b) which has to be pressed before any of these
keys:

.. code-block:: none

    ?               // List all shortcuts
    c               // Create a new window
    d               // Detatch (Exit back to normal teminal but don't quit this session)
    l               // Switch to last window
    w               // Select window from list
    0 to 9          // Switch to window x
    ( or )          // Switch to previous or next session
    p or n          // Switch to previous or next window
    % or "          // Split pane horizonally or vertically
    q and 0 to 9    // Switch to pane
    z               // Fullscreen pane
    !               // Convert pane into a window
    $               // Rename current session
    ,               // Rename current window
    :               // Enter message-box


When you are using the recommended :ref:`tmux.conf<tmux_configure>`:

.. code-block:: none

    Ctrl-hjkl       // (Without prefix) Switch to pane
    Ctrl-hjkl       // (With prfix) Resize pane

The copy mode is an easy way to navigate your history. You have to enable the Vim bindings
by putting :code:`setw -g mode-keys vi` inside your `tmux.conf` (The default are Emacs
bindings).

.. code-block:: none

    [ or PgUp       // Enter copy mode
    Space           // Start selection
    Enter           // Copy selection
    ]               // Paste from buffer_0
    #               // List all buffers
    =               // Choose which buffer to paste

.. note::

    These were only the commands and shortcuts I find the most important. There are many
    more. Take a look at this `cheatsheet <https://tmuxcheatsheet.com/>`_.
