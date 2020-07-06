Make it Your Own
================


The Config File
---------------

Start by creating a config file. The default is :code:`~/.tmux.conf` but you can create
one wherever you like.

When updating your config file you have to execute while having at least one Tmux session
active: :code:`tmux source-file /path/to/file/tmux.conf`.

When you decided not to create a tmux file at :code:`~/.tmux.conf`, you have to start Tmux
with :code:`tmux -f /path/to/file/tmux.conf` (Just create an alias!).

In order to start a new session with UTF-8 support, you have to start Tmux
with :code:`tmux -u`.

.. _tmux_configure:

Go Configure!
-------------

This is a very good start if you are also using Vim:

.. code-block:: none

    # Prefix
    set-option -g prefix2 C-a

    # Reload config
    unbind r
    bind r source-file /path/to/file/tmux.conf

    # Vim Shortcuts for copy mode
    setw -g mode-keys vi

    # Fix esc key taking longer
    set -s escape-time 0
    # Increase time for repeating shortcuts (bind-key -r)
    set -g repeat-time 1000
    # Increase the time for which the pane indicators are shown
    set -g display-panes-time 2500
    # Start page index at 1
    set -g base-index 1
    setw -g pane-base-index 1
    # Don't rename windows automatically (Rename with ,)
    set-option -g allow-rename off

    # Resize pane with Control-hjkl
    bind-key -r 'C-k' resize-pane -U 5
    bind-key -r 'C-j' resize-pane -D 5
    bind-key -r 'C-h' resize-pane -L 5
    bind-key -r 'C-l' resize-pane -R 5

    # Enable repetitive hitting next and previous for switching windows
    bind-key -r n next-window
    bind-key -r p previous-window

    ##############################################################################
    ### vim-tmux-navigator - https://github.com/christoomey/vim-tmux-navigator ###
    ##############################################################################
    is_vim="ps -o state= -o comm= -t '#{pane_tty}' \
        | grep -iqE '^[^TXZ ]+ +(\\S+\\/)?g?(view|n?vim?x?)(diff)?$'"
    bind-key -n 'C-h' if-shell "$is_vim" 'send-keys C-h'  'select-pane -L'
    bind-key -n 'C-j' if-shell "$is_vim" 'send-keys C-j'  'select-pane -D'
    bind-key -n 'C-k' if-shell "$is_vim" 'send-keys C-k'  'select-pane -U'
    bind-key -n 'C-l' if-shell "$is_vim" 'send-keys C-l'  'select-pane -R'
    tmux_version='$(tmux -V | sed -En "s/^tmux ([0-9]+(.[0-9]+)?).*/\1/p")'
    if-shell -b '[ "$(echo "$tmux_version < 3.0" | bc)" = 1 ]' \
        "bind-key -n 'C-\\' if-shell \"$is_vim\" 'send-keys C-\\'  'select-pane -l'"
    if-shell -b '[ "$(echo "$tmux_version >= 3.0" | bc)" = 1 ]' \
        "bind-key -n 'C-\\' if-shell \"$is_vim\" 'send-keys C-\\\\' 'select-pane -l'"
    bind-key -T copy-mode-vi 'C-h' select-pane -L
    bind-key -T copy-mode-vi 'C-j' select-pane -D
    bind-key -T copy-mode-vi 'C-k' select-pane -U
    bind-key -T copy-mode-vi 'C-l' select-pane -R
    bind-key -T copy-mode-vi 'C-\' select-pane -l

`Here <https://cassidy.codes/blog/2019-08-03-tmux-colour-theme/>`_ is a good article on
how to customize it visually.

For more resources on Tmux take a look `here <https://github.com/rothgar/awesome-tmux>`_.

You can also find my current config files `here <https://github.com/tobiasgreiser/dotfiles>`_.
