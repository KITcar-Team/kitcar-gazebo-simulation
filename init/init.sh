# ask for init or cleanup
read -p "do you want to initialize (1) or cleanup (2) kitcar-gazebo-simulation? " option

case "$option" in
  1) # do initialize
    # check for dependencies
    if [ -z "$KITCAR_REPO_PATH" ]; then
      echo "ERROR: can't find \$KITCAR_REPO_PATH"
      echo "please run first kitcar-init script: https://git.kitcar-team.de/kitcar/kitcar-init"
      exit 1
    fi
    
    # initialize kitcar-gazebo-simulation
    echo "add kitcar-gazebo-simulation bashrc to your bashrc"
    echo "source $KITCAR_REPO_PATH/kitcar-gazebo-simulation/init/bashrc # for kitcar-gazebo-simulation repository" >> ~/.bashrc
    
    # load changes
    echo "apply changes to current terminal ..."
    source  ~/.bashrc

    # set commit hook
    echo "setting commit hook to check formatting when creating new commit ..."
    ln -sf $KITCAR_REPO_PATH/kitcar-gazebo-simulation/check_formatting.sh $KITCAR_REPO_PATH/kitcar-gazebo-simulation/.git/hooks/pre-commit
	  chmod +x $KITCAR_REPO_PATH/kitcar-gazebo-simulation/.git/hooks/pre-commit
  ;;

  2) # do cleanup
    echo "remove kitcar-gazebo-simulation entry from bashrc"
    sed '/source .*kitcar-gazebo-simulation/d' -i ~/.bashrc

    echo "apply changes to current terminal ..."
    source  ~/.bashrc
  ;;

  *) # invalid option
    echo "invalid option, choose 1 or 2"
esac
