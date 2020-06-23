#!/bin/bash
# ask for init or cleanup
echo -e "\e[96mAttention this script installs several packages (apt and pip3)! \e[39m"
read -p "Do you want to initialize (1) or cleanup (2) kitcar-gazebo-simulation? " option

check_for_ubuntu_version(){
  grep -qE ".*(UBUNTU|VERSION)_CODENAME.*=.*${1}" /etc/os-release
  return $?
}

check_ubuntu_version(){
if check_for_ubuntu_version focal;
  then
    echo focal
    return 0
  elif check_for_ubuntu_version bionic;
  then
    echo bionic
    return 0
  else
    echo
    return 1
fi
}

case "$option" in
  1) # do initialize
    # check for dependencies
    if [ -z "$KITCAR_REPO_PATH" ]; then
      echo -e "\n\e[31mERROR: can't find \$KITCAR_REPO_PATH"
      echo -e "\e[31mplease run kitcar-init script: https://git.kitcar-team.de/kitcar/kitcar-init first\e[39m"
      exit 1
    fi

    # initialize kitcar-gazebo-simulation
    echo -e "\nadd kitcar-gazebo-simulation bashrc to your bashrc"
    echo "source $KITCAR_REPO_PATH/kitcar-gazebo-simulation/init/bashrc # for kitcar-gazebo-simulation repository" >> ~/.bashrc

    # load changes
    echo "apply changes to current terminal ..."
    source  ~/.bashrc

    UBUNTU_VERSION=$(check_ubuntu_version)
    INIT_DIR=$KITCAR_REPO_PATH/kitcar-gazebo-simulation/init

    # Install apt packages
    echo -e "\nStart installing apt packages from \e[2minit/packages_$UBUNTU_VERSION.txt\e[22m (requires sudo priviliges)"
    sudo apt-get update && sudo xargs --arg-file=$INIT_DIR/packages_$UBUNTU_VERSION.txt apt-get install -y

    # Install python packages
    echo -e "\nStart installing python packages from \e[2minit/requirements.txt\e[22m"
    case "$UBUNTU_VERSION" in
      "focal") # Ubuntu 20.04
        pip3 install --no-warn-script-location -r $INIT_DIR/requirements.txt;;
      "bionic")
        pip3 install -r $INIT_DIR/requirements.txt;;
      *)
        echo -e "\n\e[31mERROR: You are not using the correct version of Ubuntu (bionic or focal)!\e[39m" ;
        exit;
    esac
    
    source ~/.profile

    # Install pre-commit hook
    cd $KITCAR_REPO_PATH/kitcar-gazebo-simulation
    pre-commit install
  ;;

  2) # do cleanup
    echo -e "\nremove kitcar-gazebo-simulation entry from bashrc"
    sed '/source .*kitcar-gazebo-simulation/d' -i ~/.bashrc

    echo "apply changes to current terminal ..."
    # Source init / This fails when the simulation has not been built
    source  ~/.bashrc
  ;;

  *) # invalid option
    echo -e "\n\e[33minvalid option, choose 1 or 2"
esac
