#!/bin/bash

sudo apt update && sudo apt upgrade

sudo apt install -y
    bat \
    neovim \
    ripgrep \
    timg \
    vim \
    zoxide \
    zsh

# Install dotfiles
[ ! -d ~/dotfiles ] && git clone https://github.com/rafidka/dotfiles.git 
cd ~/dotfiles
./install.sh
